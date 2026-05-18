"""Pure Python deterministic UNet architecture calculations.

This module provides functions to compute UNet channel counts and parameter estimates
without requiring PyTorch or sleap-nn dependencies. Used by the training config GUI
to display model architecture information.

Ported from sleap-nn investigations:
- D:/sleap-nn/scratch/2025-12-21-deterministic-unet-channels/unet_channels.py
- D:/sleap-nn/scratch/2025-12-21-deterministic-unet-params/unet_params.py
"""

from typing import Dict, Optional
import math


def compute_unet_channels(
    filters: int = 32,
    filters_rate: float = 1.5,
    max_stride: int = 16,
    output_stride: int = 2,
    stem_stride: Optional[int] = None,
    block_contraction: bool = False,
) -> Dict[int, int]:
    """Compute UNet output channels at each stride level.

    This function deterministically computes the number of output channels
    at each stride level of a UNet architecture based on the configuration
    parameters. This matches the behavior of sleap_nn.architectures.unet.UNet.

    Args:
        filters: Base filter count. Default 32.
        filters_rate: Multiplicative factor per encoder/decoder level. Default 1.5.
        max_stride: Maximum stride of the encoder (bottleneck). Default 16.
        output_stride: Final output stride of the decoder. Default 2.
        stem_stride: Stride of the stem blocks. If None, no stem blocks. Default None.
        block_contraction: If True, reduces channels at bottleneck. Default False.

    Returns:
        Dictionary mapping stride to number of output channels at that stride.
        Keys are strides (e.g., 16, 8, 4, 2), values are channel counts.

    Example:
        >>> compute_unet_channels(
        ...     filters=32, filters_rate=1.5, max_stride=16, output_stride=2)
        {16: 162, 8: 108, 4: 72, 2: 48}
    """
    # Compute derived parameters (matches UNet.from_config)
    stem_blocks = int(math.log2(stem_stride)) if stem_stride else 0
    down_blocks = int(math.log2(max_stride)) - stem_blocks
    up_blocks = int(math.log2(max_stride / output_stride)) + stem_blocks

    stride_to_filters: Dict[int, int] = {}

    # Calculate max_stride (current_stride after encoder)
    current_stride = 2**stem_blocks if stem_blocks > 0 else 1

    # Encoder blocks contribute pooling (pool=True when block + stem_blocks > 0)
    for block in range(down_blocks):
        if block + stem_blocks > 0:
            current_stride *= 2

    # Final pool layer in encoder
    current_stride *= 2

    # Compute bottleneck channels (x_in_shape for decoder)
    if block_contraction:
        x_in_shape = int(filters * (filters_rate ** (down_blocks + stem_blocks - 1)))
    else:
        x_in_shape = int(filters * (filters_rate ** (down_blocks + stem_blocks)))

    stride_to_filters[current_stride] = x_in_shape

    # Compute decoder channels at each level
    for block in range(up_blocks):
        if block_contraction:
            block_filters_out = int(
                filters * (filters_rate ** (down_blocks + stem_blocks - 2 - block))
            )
        else:
            block_filters_out = int(
                filters
                * (filters_rate ** max(0, down_blocks + stem_blocks - 1 - block))
            )

        next_stride = current_stride // 2
        stride_to_filters[next_stride] = block_filters_out
        current_stride = next_stride

    return stride_to_filters


def _conv2d_params(
    in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True
) -> int:
    """Calculate parameters for a Conv2d layer."""
    params = in_channels * out_channels * kernel_size * kernel_size
    if bias:
        params += out_channels
    return params


def compute_unet_params(
    filters: int = 32,
    filters_rate: float = 1.5,
    max_stride: int = 16,
    output_stride: int = 2,
    stem_stride: Optional[int] = None,
    kernel_size: int = 3,
    stem_kernel_size: int = 7,
    convs_per_block: int = 2,
    middle_block: bool = True,
    up_interpolate: bool = True,
    in_channels: int = 1,
    block_contraction: bool = False,
) -> int:
    """Estimate total UNet parameter count.

    This function deterministically estimates the number of trainable parameters
    in a UNet architecture based on configuration parameters. Validated to have
    0% error against actual PyTorch model instantiation.

    Args:
        filters: Base filter count. Default 32.
        filters_rate: Multiplicative factor per level. Default 1.5.
        max_stride: Maximum stride of encoder (bottleneck). Default 16.
        output_stride: Final output stride of decoder. Default 2.
        stem_stride: Stride of stem blocks. If None, no stem. Default None.
        kernel_size: Kernel size for encoder/decoder convs. Default 3.
        stem_kernel_size: Kernel size for stem convs. Default 7.
        convs_per_block: Number of convolutions per block. Default 2.
        middle_block: Whether to include middle block. Default True.
        up_interpolate: If True, use bilinear (0 params). If False, transposed
            conv. Default True.
        in_channels: Number of input channels. Default 1.
        block_contraction: If True, reduces channels at bottleneck. Default False.

    Returns:
        Estimated total trainable parameter count.

    Example:
        >>> compute_unet_params(
        ...     filters=32, filters_rate=1.5, max_stride=16, output_stride=2)
        1295032
    """
    # Compute derived parameters
    stem_blocks = int(math.log2(stem_stride)) if stem_stride else 0
    down_blocks = int(math.log2(max_stride)) - stem_blocks
    up_blocks = int(math.log2(max_stride / output_stride)) + stem_blocks

    # Track encoder output channels at each stride for skip connections
    encoder_channels_at_stride: Dict[int, int] = {}

    total_params = 0
    current_in_channels = in_channels

    # =========================================================================
    # STEM
    # =========================================================================
    if stem_blocks > 0:
        for i in range(stem_blocks):
            block_filters = int(filters * (filters_rate**i))
            for j in range(convs_per_block):
                if j == 0:
                    conv_in = (
                        current_in_channels
                        if i == 0
                        else int(filters * (filters_rate ** (i - 1)))
                    )
                else:
                    conv_in = block_filters
                total_params += _conv2d_params(conv_in, block_filters, stem_kernel_size)
            current_in_channels = block_filters

        stem_out_channels = int(filters * (filters_rate ** (stem_blocks - 1)))
        encoder_channels_at_stride[2**stem_blocks] = stem_out_channels

    # After stem, input to encoder
    # Stem blocks affect initial encoder channels (unused in param calculation)

    # =========================================================================
    # ENCODER
    # =========================================================================
    current_stride = 2**stem_blocks if stem_blocks > 0 else 1

    for block_idx in range(down_blocks):
        block_filters = int(filters * (filters_rate ** (block_idx + stem_blocks)))
        has_pool = (block_idx + stem_blocks) > 0
        if has_pool:
            current_stride *= 2

        if block_idx == 0:
            if stem_blocks > 0:
                prev_filters = int(filters * (filters_rate ** (stem_blocks - 1)))
            else:
                prev_filters = in_channels
        else:
            prev_filters = int(
                filters * (filters_rate ** (block_idx + stem_blocks - 1))
            )

        for j in range(convs_per_block):
            conv_in = prev_filters if j == 0 else block_filters
            total_params += _conv2d_params(conv_in, block_filters, kernel_size)

        encoder_channels_at_stride[current_stride] = block_filters

    last_encoder_filters = int(
        filters * (filters_rate ** (down_blocks + stem_blocks - 1))
    )
    current_stride *= 2  # Final pool

    # =========================================================================
    # MIDDLE BLOCK
    # =========================================================================
    if middle_block:
        if convs_per_block > 1:
            expand_filters = int(
                filters * (filters_rate ** (down_blocks + stem_blocks))
            )
            for j in range(convs_per_block - 1):
                conv_in = last_encoder_filters if j == 0 else expand_filters
                total_params += _conv2d_params(conv_in, expand_filters, kernel_size)
            middle_expand_out = expand_filters
        else:
            middle_expand_out = last_encoder_filters

        if block_contraction:
            contract_filters = int(
                filters * (filters_rate ** (down_blocks + stem_blocks - 1))
            )
        else:
            contract_filters = int(
                filters * (filters_rate ** (down_blocks + stem_blocks))
            )
        total_params += _conv2d_params(middle_expand_out, contract_filters, kernel_size)

    # =========================================================================
    # DECODER
    # =========================================================================
    if block_contraction:
        decoder_in_channels = int(
            filters * (filters_rate ** (down_blocks + stem_blocks - 1))
        )
    else:
        decoder_in_channels = int(
            filters * (filters_rate ** (down_blocks + stem_blocks))
        )

    prev_block_out = decoder_in_channels

    for block_idx in range(up_blocks):
        if block_contraction:
            block_filters_out = int(
                filters * (filters_rate ** (down_blocks + stem_blocks - 2 - block_idx))
            )
        else:
            block_filters_out = int(
                filters
                * (filters_rate ** max(0, down_blocks + stem_blocks - 1 - block_idx))
            )

        next_stride = current_stride // 2
        has_skip = block_idx < down_blocks + stem_blocks

        # Transposed conv (if not using bilinear)
        if not up_interpolate:
            total_params += _conv2d_params(
                prev_block_out, block_filters_out, kernel_size
            )
            upsampled_channels = block_filters_out
        else:
            upsampled_channels = prev_block_out

        # Skip connection doubles input channels
        if has_skip:
            skip_channels = encoder_channels_at_stride.get(next_stride, 0)
            if skip_channels == 0:
                skip_channels = int(
                    filters
                    * (
                        filters_rate
                        ** max(0, down_blocks + stem_blocks - 1 - block_idx)
                    )
                )
            refine_in_channels = upsampled_channels + skip_channels
        else:
            refine_in_channels = upsampled_channels

        # Refinement convolutions
        for j in range(convs_per_block):
            conv_in = refine_in_channels if j == 0 else block_filters_out
            total_params += _conv2d_params(conv_in, block_filters_out, kernel_size)

        prev_block_out = block_filters_out
        current_stride = next_stride

    return total_params


def format_params(params: int) -> str:
    """Format parameter count with appropriate units.

    Args:
        params: Number of parameters.

    Returns:
        Human-readable string like "1.30M" or "533.4K".
    """
    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    elif params >= 1_000:
        return f"{params / 1_000:.1f}K"
    else:
        return str(params)
