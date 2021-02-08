"""Generic encoder-decoder fully convolutional backbones.

This module contains building blocks for creating encoder-decoder architectures of
general form.

The encoder branch of the network forms the initial multi-scale feature extraction via
repeated blocks of convolutions and pooling steps.

The decoder branch is then responsible for upsampling the low resolution feature maps
to achieve the target output stride.

This pattern is generalizable and describes most fully convolutional architectures. For
example:
    - simple convolutions with pooling form the structure in `LEAP CNN
<https://www.nature.com/articles/s41592-018-0234-5>`_;
    - adding skip connections forms `U-Net <https://arxiv.org/pdf/1505.04597.pdf>`_;
    - using residual blocks with skip connections forms the base module in `stacked
    hourglass <https://arxiv.org/pdf/1603.06937.pdf>`_;
    - using dense blocks with skip connections forms `FC-DenseNet
<https://arxiv.org/pdf/1611.09326.pdf>`_.

This module implements blocks used in all of these variants on top of a generic base
classes.

See the `EncoderDecoder` base class for requirements for creating new architectures.
"""

import numpy as np
import tensorflow as tf
import attr
from typing import Text, TypeVar, Sequence, Optional, Tuple, List, Union

from sleap.nn.architectures.common import IntermediateFeature


@attr.s(auto_attribs=True)
class EncoderBlock:
    """Base class for encoder blocks.

    Attributes:
        pool: If True, applies max pooling at the end of the block.
        pooling_stride: Stride of the max pooling operation. If 1, the output of this
            block will be at the same stride (== 1/scale) as the input.
    """

    pool: bool = True
    pooling_stride: int = 2

    def make_block(self, x_in: tf.Tensor) -> tf.Tensor:
        """Instantiate the encoder block from an input tensor."""
        raise NotImplementedError(
            "Subclasses of EncoderBlock must implement make_block."
        )


@attr.s(auto_attribs=True)
class SimpleConvBlock(EncoderBlock):
    """Flexible block of convolutions and max pooling.

    Attributes:
        pool: If True, applies max pooling at the end of the block.
        pooling_stride: Stride of the max pooling operation. If 1, the output of this
            block will be at the same stride (== 1/scale) as the input.
        pool_before_convs: If True, max pooling is performed before convolutions.
        num_convs: Number of convolution layers with activation. All attributes below
            are the same for all convolution layers within the block.
        filters: Number of convolutional kernel filters.
        kernel_size: Size of convolutional kernels (== height == width).
        use_bias: If False, convolution layers will not have a bias term.
        batch_norm: If True, applies batch normalization after each convolution.
        batch_norm_before_activation: If True, batch normalization is applied to the
            features computed from the linear convolution operation before the
            activation function, i.e.:
                conv -> BN -> activation function
            If False, the mini-block will look like:
                conv -> activation function -> BN
        activation: Name of activation function (typically "relu" or "linear").
        block_prefix: String to append to the prefix provided at block creation time.

    Note:
        This block is used in LeapCNN and UNet.
    """

    pool_before_convs: bool = False
    num_convs: int = 2
    filters: int = 32
    kernel_size: int = 3
    use_bias: bool = True
    batch_norm: bool = False
    batch_norm_before_activation: bool = True
    activation: Text = "relu"
    block_prefix: Text = ""

    def make_block(self, x_in: tf.Tensor, prefix: Text = "conv_block") -> tf.Tensor:
        """Create the block from an input tensor.

        Args:
            x_in: Input tensor to the block.
            prefix: String that will be added to the name of every layer in the block.
                If not specified, instantiating this block multiple times may result in
                name conflicts if existing layers have the same name.

        Returns:
            The output tensor after applying all operations in the block.
        """
        prefix += self.block_prefix
        x = x_in
        if self.pool and self.pool_before_convs:
            x = tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=self.pooling_stride,
                padding="same",
                name=f"{prefix}_pool",
            )(x)

        for i in range(self.num_convs):
            x = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                name=f"{prefix}_conv{i}",
            )(x)

            if self.batch_norm and self.batch_norm_before_activation:
                x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn{i}")(x)

            x = tf.keras.layers.Activation(
                activation=self.activation, name=f"{prefix}_act{i}_{self.activation}"
            )(x)

            if self.batch_norm and not self.batch_norm_before_activation:
                x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn{i}")(x)

        if self.pool and not self.pool_before_convs:
            x = tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=self.pooling_stride,
                padding="same",
                name=f"{prefix}_pool",
            )(x)

        return x


@attr.s(auto_attribs=True)
class DecoderBlock:
    """Base class for decoder blocks.

    Attributes:
        upsampling_stride: The striding of the upsampling layer. This is typically set
            to 2, such that the input tensor doubles in size after the block, but can be
            set higher to upsample in fewer steps.
    """

    upsampling_stride: int = 2

    def make_block(
        self,
        x: tf.Tensor,
        current_stride: Optional[int],
        skip_source: Optional[tf.Tensor] = None,
        prefix: Text = "upsample",
    ) -> tf.Tensor:
        """Instantiate the decoder block from an input tensor.

        Args:
            x_in: Input tensor to the block.
            current_stride: The stride of input tensor.
            skip_source: A tensor that will be used to form a skip connection if
                the block is configured to use it.
            prefix: String that will be added to the name of every layer in the block.
                If not specified, instantiating this block multiple times may result in
                name conflicts if existing layers have the same name.

        Returns:
            The output tensor after applying all operations in the block.
        """
        raise NotImplementedError(
            "Subclasses of DecoderBlock must implement make_block."
        )


@attr.s(auto_attribs=True)
class SimpleUpsamplingBlock(DecoderBlock):
    """Standard block of upsampling with optional refinement and skip connections.

    Attributes:
        upsampling_stride: The striding of the upsampling layer. This is typically set
            to 2, such that the input tensor doubles in size after the block, but can be
            set higher to upsample in fewer steps.
        transposed_conv: If True, use a strided transposed convolution to perform
            learnable upsampling. If False, interpolated upsampling will be used (see
            `interp_method`) and `transposed_conv_*` attributes will have no effect.
        transposed_conv_filters: Integer that specifies the number of filters in the
            transposed convolution layer.
        transposed_conv_kernel_size: Size of the kernel for the transposed convolution.
        transposed_conv_use_bias: If False, transposed convolution layers will not have
            a bias term.
        transposed_conv_batch_norm: If True, applies batch normalization after the
            transposed convolution.
        transposed_conv_batch_norm_before_activation: If True, batch normalization is
            applied to the features computed from the linear transposed convolution
            operation before the activation function, i.e.:
                transposed conv -> BN -> activation function
            If False, the mini-block will look like:
                transposed conv -> activation function -> BN
        transposed_conv_activation: Name of activation function (typically "relu" or
            "linear").
        interp_method: String specifying the type of interpolation to use if
            `transposed_conv` is set to False. This can be `bilinear` or `nearest`. See
            `tf.keras.layers.UpSampling2D` for more details on the implementation.
        skip_connection: If True, the block will form a skip connection with source
            features if provided during instantiation in the `make_block` method. If
            False, no skip connection will be formed even if a source feature is
            available.
        skip_add: If True, the skip connection will be formed by adding the source
            feature to the output of the upsampling operation. If they have different
            number of channels, a 1x1 linear convolution will be applied to the source
            first (similar to residual shortcut connections). If False, the two tensors
            will be concatenated channelwise instead.
        refine_convs: If greater than 0, specifies the number of convolutions that will
            be applied after the upsampling step. These layers can serve the purpose of
            "mixing" the skip connection fused features, or to refine the current
            feature map after upsampling which can help to prevent aliasing and
            checkerboard effects. If 0, no additional convolutions will be applied after
            upsampling and the skip connection (if present) and all `refine_convs_*`
            attributes will have no effect. If greater than 1, all layers will be
            identical with respect to these attributes.
        refine_convs_first_filters: If not None, the first refinement conv layer will
            have this many filters, otherwise `refine_convs_filters`.
        refine_convs_filters: Specifies the number of filters to use for the refinement
            convolutions.
        refine_convs_kernel_size: Size of the kernel for the refinement convolution.
        refine_convs_use_bias: If False, refinement convolution layers will not have a
            bias term.
        refine_convs_batch_norm: If True, applies batch normalization after each
            refinement convolution.
        refine_convs_batch_norm_before_activation: If True, batch normalization is
            applied to the features computed from each linear refinement convolution
            operation before the activation function, i.e.:
                conv -> BN -> activation function
            If False, the mini-block will look like:
                conv -> activation function -> BN
        refine_convs_activation: Name of activation function (typically "relu" or
            "linear").

    Note:
        This block is used in LeapCNN and UNet.
    """

    transposed_conv: bool = False
    transposed_conv_filters: int = 64
    transposed_conv_kernel_size: int = 3
    transposed_conv_use_bias: bool = True
    transposed_conv_batch_norm: bool = True
    transposed_conv_batch_norm_before_activation: bool = True
    transposed_conv_activation: Text = "relu"

    interp_method: Text = "bilinear"

    skip_connection: bool = False
    skip_add: bool = False

    refine_convs: int = 2
    refine_convs_first_filters: Optional[int] = None
    refine_convs_filters: int = 64
    refine_convs_use_bias: bool = True
    refine_convs_kernel_size: int = 3
    refine_convs_batch_norm: bool = True
    refine_convs_batch_norm_before_activation: bool = True
    refine_convs_activation: Text = "relu"

    def make_block(
        self,
        x: tf.Tensor,
        current_stride: Optional[int] = None,
        skip_source: Optional[tf.Tensor] = None,
        prefix: Text = "upsample",
    ) -> tf.Tensor:
        """Instantiate the decoder block from an input tensor.

        Args:
            x_in: Input tensor to the block.
            current_stride: The stride of input tensor. Not required but if provided,
                will be used to prepend the strides to the prefix.
            skip_source: A tensor that will be used to form a skip connection if
                the block is configured to use it.
            prefix: String that will be added to the name of every layer in the block.
                If not specified, instantiating this block multiple times may result in
                name conflicts if existing layers have the same name.

        Returns:
            The output tensor after applying all operations in the block.
        """
        if current_stride is not None:
            # Append the strides to the block prefix.
            new_stride = current_stride // self.upsampling_stride
            prefix += f"_s{current_stride}_to_s{new_stride}"

        if self.transposed_conv:
            # Upsample via strided transposed convolution.
            x = tf.keras.layers.Conv2DTranspose(
                filters=self.transposed_conv_filters,
                kernel_size=self.transposed_conv_kernel_size,
                strides=self.upsampling_stride,
                padding="same",
                name=f"{prefix}_trans_conv",
            )(x)

            if (
                self.transposed_conv_batch_norm
                and self.transposed_conv_batch_norm_before_activation
            ):
                x = tf.keras.layers.BatchNormalization(name=f"{prefix}_trans_conv_bn")(
                    x
                )

            x = tf.keras.layers.Activation(
                activation=self.transposed_conv_activation,
                name=f"{prefix}_trans_conv_act_{self.transposed_conv_activation}",
            )(x)

            if (
                self.transposed_conv_batch_norm
                and not self.transposed_conv_batch_norm_before_activation
            ):
                x = tf.keras.layers.BatchNormalization(name=f"{prefix}_trans_conv_bn")(
                    x
                )

        else:
            # Upsample via interpolation.
            x = tf.keras.layers.UpSampling2D(
                size=self.upsampling_stride,
                interpolation=self.interp_method,
                name=f"{prefix}_interp_{self.interp_method}",
            )(x)

        # Form skip connection if available.
        if self.skip_connection and skip_source is not None:
            if self.skip_add:
                source_x = skip_source
                if source_x.shape[-1] != x.shape[-1]:
                    # Adjust channel count via 1x1 linear conv if not matching.
                    source_x = tf.keras.layers.Conv2D(
                        filters=x.shape[-1],
                        kernel_size=1,
                        strides=1,
                        padding="same",
                        name=f"{prefix}_skip_conv1x1",
                    )(source_x)

                # Skip connection via addition.
                x = tf.keras.layers.Add(name=f"{prefix}_skip_add")([source_x, x])

            else:
                # Skip connection via simple concatenation.
                x = tf.keras.layers.Concatenate(name=f"{prefix}_skip_concat")(
                    [skip_source, x]
                )

        # Add further convolutions to refine after upsampling and/or skip.
        for i in range(self.refine_convs):
            filters = self.refine_convs_filters
            if i == 0 and self.refine_convs_first_filters is not None:
                filters = self.refine_convs_first_filters
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=self.refine_convs_kernel_size,
                strides=1,
                padding="same",
                use_bias=self.refine_convs_use_bias,
                name=f"{prefix}_refine_conv{i}",
            )(x)

            if (
                self.refine_convs_batch_norm
                and self.refine_convs_batch_norm_before_activation
            ):
                x = tf.keras.layers.BatchNormalization(
                    name=f"{prefix}_refine_conv{i}_bn"
                )(x)

            x = tf.keras.layers.Activation(
                activation=self.transposed_conv_activation,
                name=f"{prefix}_refine_conv{i}_act_{self.refine_convs_activation}",
            )(x)

            if (
                self.refine_convs_batch_norm
                and not self.refine_convs_batch_norm_before_activation
            ):
                x = tf.keras.layers.BatchNormalization(
                    name=f"{prefix}_refine_conv{i}_bn"
                )(x)

        return x


@attr.s(auto_attribs=True)
class EncoderDecoder:
    """General encoder-decoder base class.

    New architectures that follow the encoder-decoder pattern can be defined by
    inheriting from this class and implementing the `encoder_stack` and `decoder_stack`
    methods.

    Attributes:
        stacks: If greater than 1, the encoder-decoder architecture will be repeated.
    """

    stacks: int = 1

    @property
    def stem_stack(self) -> Optional[Sequence[EncoderBlock]]:
        """Return a list of encoder blocks that define the stem."""
        return None

    @property
    def encoder_stack(self) -> Sequence[EncoderBlock]:
        """Return a list of encoder blocks that define the encoder."""
        raise NotImplementedError(
            "Encoder-decoder subclasses must define encoder stack."
        )

    @property
    def decoder_stack(self) -> Sequence[DecoderBlock]:
        """Return a list of decoder blocks that define the decoder."""
        raise NotImplementedError(
            "Encoder-decoder subclasses must define decoder stack."
        )

    @property
    def stem_features_stride(self) -> int:
        """Return the relative stride of the final output of the stem block.

        This is equivalent to the stride of the stem assuming that it is constructed
        from an input with stride 1.
        """
        if self.stem_stack is None:
            return 1

        return int(
            np.prod([block.pooling_stride for block in self.stem_stack if block.pool])
        )

    @property
    def encoder_features_stride(self) -> int:
        """Return the relative stride of the final output of the encoder.

        This is equivalent to the stride of the encoder assuming that it is constructed
        from an input with stride 1.
        """
        return int(
            np.prod(
                [block.pooling_stride for block in self.encoder_stack if block.pool]
                + [self.stem_features_stride]
            )
        )

    @property
    def decoder_features_stride(self) -> int:
        """Return the relative stride of the final output of the decoder.

        This is equivalent to the stride of the decoder assuming that it is constructed
        from an input with stride 1.
        """
        return self.encoder_features_stride // int(
            np.prod([block.upsampling_stride for block in self.decoder_stack])
        )

    @property
    def maximum_stride(self) -> int:
        """Return the maximum stride that the input must be divisible by."""
        return self.encoder_features_stride

    @property
    def output_stride(self) -> int:
        """Return stride of the output of the backbone."""
        return self.decoder_features_stride

    def make_stem(self, x_in: tf.Tensor, prefix: Text = "stem") -> tf.Tensor:
        """Instantiate the stem layers defined by the stem block configuration.

        Unlike in the encoder, the stem layers do not get repeated in stacked models.

        Args:
            x_in: The input tensor.
            current_stride: The stride of `x_in` relative to the original input. If any
                pooling was performed before the stem, this must be specified in
                order to appropriately set the stride in the rest of the model.
            prefix: String prefix for naming stem layers.

        Returns:
            The final output tensor of the stem.
        """
        if self.stem_stack is None:
            return x_in

        x = x_in
        for i, block in enumerate(self.stem_stack):
            # Instantiate block.
            x = block.make_block(x, prefix=f"{prefix}{i}")
        return x

    def make_encoder(
        self, x_in: tf.Tensor, current_stride: int, prefix: Text = "enc"
    ) -> Tuple[tf.Tensor, List[IntermediateFeature]]:
        """Instantiate the encoder layers defined by the encoder stack configuration.

        Args:
            x_in: The input tensor.
            current_stride: The stride of `x_in` relative to the original input. If any
                pooling was performed before the encoder, this must be specified in
                order to appropriately set the stride in the returned intermediate
                features.
            prefix: String prefix for naming encoder layers.

        Returns:
            A tuple of the final output tensor of the encoder and a list of
            `IntermediateFeature`s.

            The intermediate features contain the output tensors from every block except
            the last. These can be reused in the decoder to form skip connections.
        """
        x = x_in
        intermediate_features = []
        for i, block in enumerate(self.encoder_stack):

            # Instantiate block.
            x = block.make_block(x, prefix=f"{prefix}{i}")

            # Update the current stride and store the output of the current block.
            if block.pool:
                current_stride *= block.pooling_stride

            if current_stride not in [feat.stride for feat in intermediate_features]:
                intermediate_features.append(
                    IntermediateFeature(tensor=x, stride=current_stride)
                )

        return x, intermediate_features[:-1]

    def make_decoder(
        self,
        x_in: tf.Tensor,
        current_stride: int,
        skip_source_features: Optional[Sequence[IntermediateFeature]] = None,
        prefix: Text = "dec",
    ) -> Tuple[tf.Tensor, List[IntermediateFeature]]:
        """Instantiate the encoder layers defined by the decoder stack configuration.

        Args:
            x_in: The input tensor.
            current_stride: The stride of `x_in` relative to the original input. This is
                the stride of the output of the encoder relative to the original input.
            skip_source_features: A sequence of `IntermediateFeature`s containing
                tensors that can be used to form skip connections at matching strides.
                At every decoder block, the first skip source feature found at the input
                stride of the block will be passed to the block instantiation method. If
                the decoder block is not configured to form skip connections, these will
                be ignored even if found.
            prefix: String prefix for naming decoder layers.

        Returns:
            A tuple of the final output tensor of the decoder and a list of
            `IntermediateFeature`s.

            The intermediate features contain the output tensors from every block except
            the last. This includes the input to this function (`x_in`). These are
            useful when defining heads that take inputs at multiple scales.
        """
        x = x_in
        intermediate_features = []
        for i, block in enumerate(self.decoder_stack):

            # Store the output of the current block.
            intermediate_features.append(
                IntermediateFeature(tensor=x, stride=current_stride)
            )

            next_stride = current_stride // block.upsampling_stride

            # Look for a source tensor at the next stride (after upsampling) to form a
            # skip connection.
            skip_source = None
            for source_feat in skip_source_features:
                if source_feat.stride == next_stride:
                    skip_source = source_feat.tensor
                    break

            # Create the block.
            x = block.make_block(
                x,
                current_stride=current_stride,
                skip_source=skip_source,
                prefix=f"{prefix}{i}",
            )

            # Update current stride.
            current_stride = next_stride

        return x, intermediate_features

    def make_backbone(
        self, x_in: tf.Tensor, current_stride: int = 1
    ) -> Union[
        Tuple[tf.Tensor, List[IntermediateFeature]],
        Tuple[List[tf.Tensor], List[List[IntermediateFeature]]],
    ]:
        """Instantiate the entire encoder-decoder backbone.

        Args:
            x_in: The input tensor.
            current_stride: The stride of `x_in` relative to the original input. This is
                1 if the input tensor comes from the input layer of the network. If not,
                this must be set appropriately in order to match up intermediate tensors
                during decoder construction.

        Returns:
            A tuple of the final output tensor of the decoder and a list of
            `IntermediateFeature`s.

            The intermediate features contain the output tensors from every block except
            the last. This includes the input to this function (`x_in`). These are
            useful when defining heads that take inputs at multiple scales.

            If the architecture has more than 1 stack, the outputs are each lists of
            output tensors and intermediate features corresponding to each stack.
        """
        if self.stacks > 1:
            if self.stem_features_stride != self.decoder_features_stride:
                raise ValueError(
                    "If using a stacked configuration, the backbone must define "
                    "symmetric encoder and decoder. Create a stem for initial "
                    "downsampling if an output stride > 1 is desired."
                )

        # Build stem for the first stack if defined.
        x = self.make_stem(x_in, prefix="stem")
        stem_output = []
        if self.stem_stack is not None:
            stem_output = [
                IntermediateFeature(
                    tensor=x, stride=current_stride * self.stem_features_stride
                )
            ]

        stack_outputs = []
        intermediate_outputs = []
        for i in range(self.stacks):

            # Build encoder.
            x, intermediate_encoder_features = self.make_encoder(
                x,
                current_stride=current_stride * self.stem_features_stride,
                prefix=f"stack{i}_enc",
            )

            # Build decoder.
            x, intermediate_decoder_features = self.make_decoder(
                x,
                skip_source_features=stem_output + intermediate_encoder_features,
                current_stride=current_stride * self.encoder_features_stride,
                prefix=f"stack{i}_dec",
            )

            stack_outputs.append(x)
            intermediate_outputs.append(intermediate_decoder_features)

        if self.stacks == 1:
            return stack_outputs[0], intermediate_outputs[0]
        else:
            return stack_outputs, intermediate_outputs
