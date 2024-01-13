import torch
import torch.nn.functional as F

from elasticai.creator.nn import Sequential as TranslatableSequential
from elasticai.creator.nn.fixed_point import (
    quantize,
    BatchNormedConv1d,
    Linear,
    HardTanh,
)


class ProbabilityMapSoftmax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *first_dims, height, width = inputs.shape
        logit_vector = inputs.view(*first_dims, height * width)
        probability_vector = F.softmax(logit_vector, dim=len(first_dims))
        probability_map = probability_vector.view(*first_dims, height, width)
        return probability_map


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

        self.hardware_model = TranslatableSequential(
            BatchNormedConv1d(
                in_channels=3,
                out_channels=16,
                signal_length=signal_length,
                kernel_size=32,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            BatchNormedConv1d(
                in_channels=16,
                out_channels=8,
                signal_length=signal_length - (32 - 1),
                kernel_size=64,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            BatchNormedConv1d(
                in_channels=8,
                out_channels=4,
                signal_length=signal_length - (32 - 1) - (64 - 1),
                kernel_size=128,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            BatchNormedConv1d(
                in_channels=4,
                out_channels=2,
                signal_length=signal_length - (32 - 1) - (64 - 1) - (128 - 1),
                kernel_size=256,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            BatchNormedConv1d(
                in_channels=2,
                out_channels=1,
                signal_length=signal_length
                - (32 - 1)
                - (64 - 1)
                - (128 - 1)
                - (256 - 1),
                kernel_size=512,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            Linear(
                in_features=signal_length
                - (32 - 1)
                - (64 - 1)
                - (128 - 1)
                - (256 - 1)
                - (512 - 1),
                out_features=self.grid_width * self.grid_height,
                bias=True,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
        )
        # self.softmax = ProbabilityMapSoftmax()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *first_dims, in_channels, signal_length = inputs.shape
        quantized_inputs = quantize(
            x=inputs.view(*first_dims, in_channels * signal_length),
            total_bits=self.total_bits,
            frac_bits=self.frac_bits,
        )
        predictions = self.hardware_model(quantized_inputs)
        return predictions
        # return  self.softmax(logit_map)
