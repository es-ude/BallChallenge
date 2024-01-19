import torch
from torch.nn import Flatten as _Flatten
from elasticai.creator.nn import Sequential as TranslatableSequential
from elasticai.creator.nn.fixed_point import (
    quantize,
    Conv1d,
    Linear,
    HardTanh,
)
from elasticai.creator.vhdl.shared_designs.null_design import NullDesign


class Flatten(_Flatten):
    def create_design(self, name: str):
        return NullDesign(name)


class BallChallengeModel(torch.nn.Module):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        signal_length: int,
        grid_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.grid_width, self.grid_height = grid_size
        self.total_bits = total_bits
        self.frac_bits = frac_bits
        conv = Conv1d
        self.hardware_model = TranslatableSequential(
            conv(
                in_channels=3,
                out_channels=16,
                signal_length=signal_length,
                kernel_size=8,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            conv(
                in_channels=16,
                out_channels=8,
                signal_length=signal_length - (8 - 1),
                kernel_size=16,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            conv(
                in_channels=8,
                out_channels=4,
                signal_length=signal_length - (8 - 1) - (16 - 1),
                kernel_size=32,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            conv(
                in_channels=4,
                out_channels=2,
                signal_length=signal_length - (8 - 1) - (16 - 1) - (32 - 1),
                kernel_size=64,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            conv(
                in_channels=2,
                out_channels=1,
                signal_length=signal_length - (8 - 1) - (16 - 1) - (32 - 1) - (64 - 1),
                kernel_size=128,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            Flatten(),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            Linear(
                in_features=signal_length
                - (8 - 1)
                - (16 - 1)
                - (32 - 1)
                - (64 - 1)
                - (128 - 1),
                out_features=self.grid_width * self.grid_height,
                bias=True,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_inputs = quantize(
            x=inputs,
            total_bits=self.total_bits,
            frac_bits=self.frac_bits,
        )
        predictions = self.hardware_model(quantized_inputs)
        return predictions
