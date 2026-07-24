"""Configurable NCSN++ score model for complex STFT conditioning."""

from typing import Any, Dict, List, Sequence

import torch
from torch import nn

from .blocks import (
    AttentionBlock,
    BigGANResBlock,
    FIRDownsample2d,
    FIRUpsample2d,
    GaussianFourierProjection,
    align_spatial,
    group_count,
)


class NCSNpp(nn.Module):
    """Rectangular-spectrogram NCSN++ with progressive input/output.

    Input is ``[B,4,F,N]`` ordered as perturbed real/imag and condition
    real/imag. Output is score channels ``[B,2,F,N]``.
    """

    def __init__(
        self,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 1, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        image_size: int = 256,
        dropout: float = 0.0,
        fourier_scale: float = 16.0,
        fir_kernel: Sequence[float] = (1, 3, 3, 1),
        skip_rescale: bool = True,
        progressive_input: bool = True,
        progressive_output: bool = True,
        input_channels: int = 4,
        output_channels: int = 2,
    ) -> None:
        super().__init__()
        if input_channels != 4 or output_channels != 2:
            raise ValueError("Complex conditional score model requires 4 input and 2 output channels")
        if num_res_blocks < 1 or not channel_multipliers:
            raise ValueError("NCSN++ requires at least one level and residual block")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.levels = len(channel_multipliers)
        self.progressive_input = bool(progressive_input)
        self.progressive_output = bool(progressive_output)
        self.attention_resolutions = set(int(value) for value in attention_resolutions)
        self.level_resolutions = [
            max(1, int(image_size) // (2 ** level)) for level in range(self.levels)
        ]

        embedding_size = int(base_channels)
        self.fourier = GaussianFourierProjection(embedding_size, fourier_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, 4 * base_channels),
            nn.SiLU(),
            nn.Linear(4 * base_channels, 4 * base_channels),
        )
        time_channels = 4 * base_channels
        channels = [int(base_channels * multiplier) for multiplier in channel_multipliers]
        self.input_conv = nn.Conv2d(input_channels, channels[0], 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attention = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.input_downsamplers = nn.ModuleList()
        self.input_projections = nn.ModuleList()
        current = channels[0]
        for level, out_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for block_index in range(int(num_res_blocks)):
                blocks.append(
                    BigGANResBlock(
                        current if block_index == 0 else out_channels,
                        out_channels,
                        time_channels,
                        dropout,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
            self.encoder_blocks.append(blocks)
            self.encoder_attention.append(
                AttentionBlock(out_channels, skip_rescale)
                if self.level_resolutions[level] in self.attention_resolutions
                else nn.Identity()
            )
            current = out_channels
            if level < self.levels - 1:
                next_channels = channels[level + 1]
                self.down_blocks.append(
                    BigGANResBlock(
                        current,
                        next_channels,
                        time_channels,
                        dropout,
                        down=True,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
                self.input_downsamplers.append(FIRDownsample2d(fir_kernel))
                self.input_projections.append(
                    nn.Conv2d(input_channels, next_channels, 1)
                )
                current = next_channels

        self.middle1 = BigGANResBlock(
            current, current, time_channels, dropout, fir_kernel=fir_kernel, skip_rescale=skip_rescale
        )
        self.middle_attention = AttentionBlock(current, skip_rescale)
        self.middle2 = BigGANResBlock(
            current, current, time_channels, dropout, fir_kernel=fir_kernel, skip_rescale=skip_rescale
        )

        self.decoder_levels: List[int] = list(reversed(range(self.levels)))
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        self.output_heads = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level in self.decoder_levels:
            out_channels = channels[level]
            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        BigGANResBlock(
                            current + out_channels,
                            out_channels,
                            time_channels,
                            dropout,
                            fir_kernel=fir_kernel,
                            skip_rescale=skip_rescale,
                        ),
                        BigGANResBlock(
                            out_channels,
                            out_channels,
                            time_channels,
                            dropout,
                            fir_kernel=fir_kernel,
                            skip_rescale=skip_rescale,
                        ),
                    ]
                )
            )
            self.decoder_attention.append(
                AttentionBlock(out_channels, skip_rescale)
                if self.level_resolutions[level] in self.attention_resolutions
                else nn.Identity()
            )
            self.output_heads.append(
                nn.Sequential(
                    nn.GroupNorm(group_count(out_channels), out_channels, eps=1.0e-6),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, output_channels, 3, padding=1),
                )
            )
            nn.init.zeros_(self.output_heads[-1][-1].weight)
            nn.init.zeros_(self.output_heads[-1][-1].bias)
            current = out_channels
            if level > 0:
                previous_channels = channels[level - 1]
                self.up_blocks.append(
                    BigGANResBlock(
                        current,
                        previous_channels,
                        time_channels,
                        dropout,
                        up=True,
                        fir_kernel=fir_kernel,
                        skip_rescale=skip_rescale,
                    )
                )
                current = previous_channels
        self.pyramid_upsample = FIRUpsample2d(fir_kernel)

    def forward(self, inputs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict score channels for ``inputs [B,4,F,N]`` and ``time [B]``."""
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError("NCSN++ input must have shape [B,4,F,N]")
        if time.ndim != 1 or time.shape[0] != inputs.shape[0]:
            raise ValueError("time must have shape [B]")
        original = inputs
        time_embedding = self.time_mlp(
            self.fourier(torch.log(time.float().clamp_min(1.0e-5)))
        )
        hidden = self.input_conv(inputs)
        input_pyramid = inputs
        skips = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                hidden = block(hidden, time_embedding)
            hidden = self.encoder_attention[level](hidden)
            skips.append(hidden)
            if level < self.levels - 1:
                hidden = self.down_blocks[level](hidden, time_embedding)
                if self.progressive_input:
                    input_pyramid = self.input_downsamplers[level](input_pyramid)
                    projected = self.input_projections[level](input_pyramid)
                    projected = align_spatial(projected, hidden)
                    hidden = (hidden + projected) / (2.0 ** 0.5)

        hidden = self.middle1(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle2(hidden, time_embedding)

        output_pyramid = None
        up_index = 0
        for decoder_index, level in enumerate(self.decoder_levels):
            skip = skips[level]
            hidden = align_spatial(hidden, skip)
            hidden = torch.cat((hidden, skip), dim=1)
            for block in self.decoder_blocks[decoder_index]:
                hidden = block(hidden, time_embedding)
            hidden = self.decoder_attention[decoder_index](hidden)
            head = self.output_heads[decoder_index](hidden)
            if output_pyramid is None:
                output_pyramid = head
            elif self.progressive_output:
                output_pyramid = align_spatial(
                    self.pyramid_upsample(output_pyramid), head
                ) + head
            else:
                output_pyramid = head
            if level > 0:
                hidden = self.up_blocks[up_index](hidden, time_embedding)
                up_index += 1
        output_pyramid = align_spatial(output_pyramid, original)
        return output_pyramid

    def parameter_count(self) -> int:
        """Return total model parameter count."""
        return sum(parameter.numel() for parameter in self.parameters())


def build_score_model(config: Dict[str, Any]) -> NCSNpp:
    """Build NCSN++ from the ``model`` configuration section."""
    values = dict(config["model"])
    name = values.pop("name", "ncsnpp")
    if name != "ncsnpp":
        raise ValueError("Unknown score model: {}".format(name))
    return NCSNpp(**values)

