"""(Higher)HRNet backbone.

This implementation is based on the PyTorch implementation of HRNet, modified to
implement HigherHRNet's configuration and new deconvolution heads.

Refs:
https://arxiv.org/pdf/1902.09212.pdf
https://arxiv.org/pdf/1908.10357.pdf
"""

import tensorflow as tf
import attr
from typing import List, Text, Union


def adjust_prefix(name_prefix):
    """Adds a delimiter if the prefix is not empty."""

    if name_prefix is None or len(name_prefix) == 0:
        name_prefix = ""
    else:
        if name_prefix[-1] != ".":
            name_prefix = name_prefix + "."

    return name_prefix


def conv_block(
    x_in,
    filters,
    kernel_size=3,
    stride=1,
    bias=False,
    with_batch_norm=True,
    activation="relu",
    name="conv",
    name_prefix=None,
):
    name_prefix = adjust_prefix(name_prefix)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        use_bias=False,
        padding="same",
        name=name_prefix + name,
    )(x_in)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization(name=name_prefix + name + ".bn")(x)

    if activation is not None:
        x = tf.keras.layers.Activation(
            activation, name=name_prefix + name + "." + activation
        )(x)

    return x


def simple_block(x_in, filters, stride=1, downsampling_layer=None, name_prefix=None):
    """Creates a basic residual convolutional block."""

    name_prefix = adjust_prefix(name_prefix)

    x = x_in

    # Sub-block 1
    x = conv_block(x, filters, kernel_size=3, name="conv1", name_prefix=name_prefix)

    # Sub-block 2
    x = conv_block(
        x,
        filters,
        kernel_size=3,
        activation=None,
        name="conv2",
        name_prefix=name_prefix,
    )

    # Increase the number of filters in the input if needed.
    if x_in.shape[-1] != x.shape[-1]:
        x_in = conv_block(
            x_in,
            filters=x.shape[-1],
            kernel_size=1,
            activation=None,
            name="conv_residual",
            name_prefix=name_prefix,
        )

    # Add residual and output result with non-linearity.
    x = tf.keras.layers.Add(name=name_prefix + "add_residual")([x_in, x])
    x = tf.keras.layers.Activation("relu", name=name_prefix + "relu_out")(x)

    return x


def bottleneck_block(x_in, filters, expansion_rate=4, name_prefix=None):
    """Creates a convolutional block with bottleneck."""

    name_prefix = adjust_prefix(name_prefix)

    x = x_in

    # Initial 1x1 conv.
    x = conv_block(x, filters, kernel_size=1, name="conv_in", name_prefix=name_prefix)

    # Middle 3x3 conv.
    x = conv_block(x, filters, kernel_size=3, name="conv_3x3", name_prefix=name_prefix)

    # Channel expansion with 1x1 conv.
    x = conv_block(
        x,
        filters,
        kernel_size=1,
        activation=None,
        name="conv_expand",
        name_prefix=name_prefix,
    )

    # Increase the number of filters in the input if needed.
    if x_in.shape[-1] != x.shape[-1]:
        x_in = conv_block(
            x_in,
            filters=x.shape[-1],
            kernel_size=1,
            activation=None,
            name="conv_residual",
            name_prefix=name_prefix,
        )

    # Add residual and output result with non-linearity.
    x = tf.keras.layers.Add(name=name_prefix + "add_residual")([x_in, x])
    x = tf.keras.layers.Activation("relu", name=name_prefix + "relu_out")(x)

    return x


def downsampling_block(
    x_in, down_steps, output_filters, relu_before_output=True, name_prefix=None
):
    name_prefix = adjust_prefix(name_prefix)

    intermediate_activation = None
    if relu_before_output:
        intermediate_activation = "relu"

    x = x_in
    input_filters = x.shape[-1]
    for step in range(down_steps - 1):
        # In the intermediate downsampling steps we don't change the number of filters.
        x = conv_block(
            x,
            filters=input_filters,
            stride=2,
            activation=intermediate_activation,
            name="strided_conv",
            name_prefix=f"{name_prefix}down{step + 1}",
        )

    x = conv_block(
        x,
        filters=output_filters,
        stride=2,
        name="strided_conv",
        name_prefix=f"{name_prefix}down{down_steps}",
    )

    return x


def upsampling_block(
    x_in, up_steps, output_filters, interp_method="nearest", name_prefix=None
):
    name_prefix = adjust_prefix(name_prefix)

    x = x_in
    x = conv_block(
        x,
        filters=output_filters,
        kernel_size=1,
        activation=None,
        name="conv_1x1",
        name_prefix=f"{name_prefix}up{up_steps}",
    )
    x = tf.keras.layers.UpSampling2D(
        size=2 ** up_steps,
        interpolation=interp_method,
        name=f"{name_prefix}up{up_steps}.{interp_method}",
    )(x)

    return x


def deconv_block(inputs, filters=256, kernel_size=4, strides=2, name_prefix=None):
    name_prefix = adjust_prefix(name_prefix)

    x = tf.keras.layers.Concatenate(name=name_prefix + "concat_in")(inputs)
    x = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=2,
        padding="same",
        use_bias=False,
        name=name_prefix + "deconv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=name_prefix + "bn")(x)
    x = tf.keras.layers.Activation("relu", name=name_prefix + "relu")(x)

    return x


def make_transition_layers(
    source_outputs: Union[tf.Tensor, List[tf.Tensor]],
    targets_filters: List[int],
    name="transition",
) -> List[tf.Tensor]:

    if isinstance(source_outputs, tf.Tensor):
        source_outputs = [source_outputs]

    targets_inputs = []
    for target_down_steps, target_filters in enumerate(targets_filters):
        if target_down_steps < len(source_outputs):
            # There exists a source output at the corresponding target scale.
            source_output = source_outputs[target_down_steps]

            if source_output.shape[-1] == target_filters:
                # The number of filters match up, just pass it through.
                targets_inputs.append(source_output)

            else:
                # Source and target have different number of filters, adjust with a conv
                # block.
                targets_inputs.append(
                    conv_block(
                        source_output,
                        target_filters,
                        name=f"conv_at_{target_down_steps}",
                        name_prefix=name,
                    )
                )

        else:
            # The source has fewer outputs than the target, so we'll downsample from the
            # smallest source available.
            source_output = source_outputs[-1]
            source_down_steps = len(source_outputs) - 1

            targets_inputs.append(
                downsampling_block(
                    x_in=source_output,
                    down_steps=target_down_steps - source_down_steps,
                    output_filters=target_filters,
                    relu_before_output=True,
                    name_prefix=(
                        f"{name}.downsamp_{source_down_steps}_to_{target_down_steps}"
                    ),
                )
            )

    return targets_inputs


def make_branch(x_in, block_filters=64, blocks=4, bottleneck=True, name_prefix=None):
    name_prefix = adjust_prefix(name_prefix)

    x = x_in

    for block in range(blocks):
        if bottleneck:
            x = bottleneck_block(
                x, filters=block_filters, name_prefix=f"{name_prefix}block{block + 1}"
            )
        else:
            x = simple_block(
                x, filters=block_filters, name_prefix=f"{name_prefix}block{block + 1}"
            )

    return x


def make_fuse_layers(branches_outputs, single_scale_output=False, name_prefix=None):
    name_prefix = adjust_prefix(name_prefix)

    if len(branches_outputs) == 1:
        raise ValueError("Must have more than 1 branch to create fuse layers.")

    n_fused_outputs = len(branches_outputs)

    if single_scale_output:
        # When doing single scale output, all outputs will be fused into the largest
        # scale.
        n_fused_outputs = 1

    fused_outputs = []
    for i, target_output in enumerate(branches_outputs[:n_fused_outputs]):
        # Target scale is 1 / (2 ** i)
        target_filters = target_output.shape[-1]

        # We start with the output at the current scale.
        fused_output_i = target_output

        for j, source_output in enumerate(branches_outputs):
            # Source scale is 1 / (2 ** j)

            if j > i:
                # Branch j has a smaller scale than i: Upsample to scale i.
                source_output = upsampling_block(
                    x_in=source_output,
                    up_steps=j - i,
                    output_filters=target_filters,
                    name_prefix=f"{name_prefix}fuse_{j + 1}_to_{i + 1}",
                )

            elif j == i:
                # No need to add the target to itself.
                continue

            elif j < i:
                # Branch j has a larger scale than i: downsample to scale i.
                source_output = downsampling_block(
                    x_in=source_output,
                    down_steps=i - j,
                    output_filters=target_filters,
                    relu_before_output=False,
                    name_prefix=f"{name_prefix}fuse_{j + 1}_to_{i + 1}",
                )

            # Add (scale adjusted) source to target output.
            fused_output_i = tf.keras.layers.Add(
                name=f"{name_prefix}fuse_{j + 1}_to_{i + 1}.add"
            )([fused_output_i, source_output])

        # Fused output at the current scale now has every other branch added to it.
        # Throw a nonlinearity on and add to the list of final outputs.
        fused_output_i = tf.keras.layers.Activation(
            "relu", name=f"{name_prefix}fused_{i + 1}.relu"
        )(fused_output_i)
        fused_outputs.append(fused_output_i)

    return fused_outputs


def make_stage(
    source_outputs: List[tf.Tensor],
    branches_filters: List[int],
    modules: int = 1,
    branches_blocks: Union[int, List[int]] = 4,
    bottleneck: bool = True,
    single_scale_output: bool = False,
    name: Text = "stage",
) -> List[tf.Tensor]:

    # Make transition layers for the inputs.
    branches_inputs = make_transition_layers(
        source_outputs, branches_filters, name=f"{name}.transition"
    )

    if isinstance(branches_blocks, int):
        # Convert scalar specification of the number of blocks per branch to list of
        # matching length.
        branches_blocks = [branches_blocks] * len(branches_filters)

    for module in range(modules):

        # Make the branches for this stage -> module.
        branches_outputs = []
        for i, (branch_input, branch_filters, branch_blocks) in enumerate(
            zip(branches_inputs, branches_filters, branches_blocks)
        ):
            branches_outputs.append(
                make_branch(
                    branch_input,
                    block_filters=branch_filters,
                    blocks=branch_blocks,
                    bottleneck=bottleneck,
                    name_prefix=f"{name}.module{module + 1}.branch{i + 1}",
                )
            )

        if len(branches_outputs) > 1:
            # If we have more than one branch, we'll fuse each of the outputs to each
            # other before returning.
            module_single_scale_output = single_scale_output
            if module < (modules - 1):
                module_single_scale_output = False
            branches_outputs = make_fuse_layers(
                branches_outputs,
                single_scale_output=module_single_scale_output,
                name_prefix=f"{name}.module{module + 1}",
            )

        branches_inputs = branches_outputs

    return branches_outputs


def make_deconv_module(
    inputs,
    output_filters,
    output_name,
    deconv_filters=256,
    bottleneck=True,
    deconv_residual_blocks=4,
    deconv_residual_block_filters=32,
    bilinear_upsampling=False,
    name_prefix=None,
):
    name_prefix = adjust_prefix(name_prefix)

    if bilinear_upsampling:
        x = tf.keras.layers.Concatenate(name=name_prefix + "concat_in")(inputs)
        deconv_feats = tf.keras.layers.UpSampling2D(
            interpolation="bilinear", name=name_prefix + "upsample"
        )(x)

    else:
        deconv_feats = deconv_block(
            inputs,
            filters=deconv_filters,
            kernel_size=4,
            strides=2,
            name_prefix=name_prefix,
        )

    x = deconv_feats
    for block in range(deconv_residual_blocks):
        if bottleneck:
            x = bottleneck_block(
                x,
                filters=deconv_residual_block_filters,
                name_prefix=f"{name_prefix}block{block + 1}",
            )
        else:
            x = simple_block(
                x,
                filters=deconv_residual_block_filters,
                name_prefix=f"{name_prefix}block{block + 1}",
            )

    x = conv_block(
        x,
        output_filters,
        kernel_size=1,
        with_batch_norm=False,
        activation=None,
        name=name_prefix + output_name,
    )

    return deconv_feats, x


def make_stem(x_in, filters=64, downsampling_steps=2):
    x = x_in

    for step in range(downsampling_steps):
        x = conv_block(
            x,
            filters,
            stride=2,
            activation=None,
            name=f"strided_conv{step + 1}",
            name_prefix="stem",
        )

    x = tf.keras.layers.Activation("relu", name="stem.relu_out")(x)

    return x


def make_first_stage(
    x_in, bottleneck=True, block_filters=64, blocks=4, output_filters=32
):
    x = x_in

    for block in range(blocks):
        if bottleneck:
            x = bottleneck_block(
                x, filters=block_filters, name_prefix=f"stage1.block{block + 1}"
            )
        else:
            x = simple_block(
                x, filters=block_filters, name_prefix=f"stage1.block{block + 1}"
            )

    x = conv_block(
        x, output_filters, activation=None, name="conv_out", name_prefix="stage1"
    )

    return x


def make_hrnet_backbone(
    x_in, C=32, initial_downsampling_steps=2, stem_filters=64, bottleneck=False
):

    x = make_stem(
        x_in, filters=stem_filters, downsampling_steps=initial_downsampling_steps
    )

    x = make_first_stage(
        x, bottleneck=False, block_filters=64, blocks=4, output_filters=C
    )
    x = make_stage(
        x, branches_filters=[C, C * 2], modules=1, bottleneck=bottleneck, name="stage2"
    )
    x = make_stage(
        x,
        branches_filters=[C, C * 2, C * 4],
        modules=4,
        bottleneck=bottleneck,
        name="stage3",
    )
    x = make_stage(
        x,
        branches_filters=[C, C * 2, C * 4, C * 8],
        modules=3,
        bottleneck=bottleneck,
        single_scale_output=True,
        name="stage4",
    )

    return tf.keras.Model(x_in, x[0], name=f"HRNet{C}")


def make_higher_hrnet_heads(
    hrnet_backbone,
    n_output_channels,
    n_deconv_modules,
    bottleneck=False,
    deconv_filters=256,
    bilinear_upsampling=False,
):

    backbone_feats = hrnet_backbone.output

    # Output at 1/4 resolution (by default):
    output_small = conv_block(
        backbone_feats,
        n_output_channels,
        kernel_size=1,
        with_batch_norm=False,
        activation=None,
        name="output_small",
    )

    all_feats = [backbone_feats]
    outputs = [output_small]

    for deconv_module in range(n_deconv_modules):
        deconv_inputs = [all_feats[-1], outputs[-1]]
        feats, output = make_deconv_module(
            deconv_inputs,
            output_filters=n_output_channels,
            bottleneck=bottleneck,
            deconv_filters=deconv_filters,
            bilinear_upsampling=bilinear_upsampling,
            output_name="deconv_output",
            name_prefix=f"deconv_module{deconv_module + 1}",
        )
        all_feats.append(feats)
        outputs.append(output)

    model = tf.keras.Model(
        inputs=hrnet_backbone.input,
        outputs=outputs,
        name=f"{hrnet_backbone.name}.deconv{n_deconv_modules}",
    )

    return model


@attr.s(auto_attribs=True)
class HigherHRNet:
    """HigherHRNet backbone.

    Attributes:
        C: The variant of HRNet to use. The most common is HRNet32, which has ~30M
            params. This number is effectively the number of filters at the highest
            resolution output.
        initial_downsampling_steps: Number of initial downsampling steps at the stem.
            Decrease if this introduces too much loss of resolution from the initial
            images.
        n_deconv_modules: Number of upsampling steps to perform at the head. If this is
            equal to initial_downsampling_steps, the output will be at the same scale as
            the input.
        bottleneck: If True, uses bottleneck blocks instead of simple residual blocks.
        deconv_filters: Number of filters to use in deconv blocks if using transposed
            convolutions.
        bilinear_upsampling: Use bilinear upsampling instead of transposed convolutions
            at the output heads.
    """

    C: int = 18
    initial_downsampling_steps: int = 1
    n_deconv_modules: int = 1
    bottleneck: bool = False
    deconv_filters: int = 256
    bilinear_upsampling: bool = False
    stem_filters: int = 64

    def output(self, x_in, n_output_channels):
        """Builds the layers for this backbone and return the output tensor.

        Args:
            x_in: Input 4D tf.Tensor.
            n_output_channels: The number of final output channels.

        Returns:
            higher_hrnet_model: A tf.keras.model whose outputs are a list of tf.Tensors
                at each scale of the deconv_modules.
        """

        hrnet_backbone = make_hrnet_backbone(
            x_in,
            C=self.C,
            initial_downsampling_steps=self.initial_downsampling_steps,
            bottleneck=self.bottleneck,
            stem_filters=self.stem_filters,
        )

        higher_hrnet_model = make_higher_hrnet_heads(
            hrnet_backbone,
            n_output_channels,
            self.n_deconv_modules,
            bottleneck=self.bottleneck,
            deconv_filters=self.deconv_filters,
            bilinear_upsampling=self.bilinear_upsampling,
        )

        return higher_hrnet_model

    @property
    def down_blocks(self):
        """Returns the number of downsampling steps in the model."""

        return self.initial_downsampling_steps + 3

    @property
    def output_scale(self):
        """Returns relative scaling factor of this backbone."""

        return 1 / (2 ** (self.initial_downsampling_steps - self.n_deconv_modules))
