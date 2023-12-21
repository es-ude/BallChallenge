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
        vector = inputs.view(*first_dims, height * width)
        probability_vector = F.softmax(vector, dim=len(first_dims))
        probability_map = probability_vector.view(*first_dims, height, width)
        return probability_map


class BallChallengeModel(torch.nn.Module):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        signal_length: int,
        impact_grid_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.grid_width, self.grid_height = impact_grid_size
        self.total_bits = total_bits
        self.frac_bits = frac_bits

        self.hardware_model = TranslatableSequential(
            BatchNormedConv1d(
                in_channels=3,
                out_channels=32,
                signal_length=signal_length,
                kernel_size=8,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            BatchNormedConv1d(
                in_channels=32,
                out_channels=4,
                signal_length=signal_length - 7,
                kernel_size=4,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
            HardTanh(total_bits=total_bits, frac_bits=frac_bits),
            Linear(
                in_features=(signal_length - 10) * 4,
                out_features=self.grid_width * self.grid_height,
                bias=True,
                total_bits=total_bits,
                frac_bits=frac_bits,
            ),
        )
        self.softmax = ProbabilityMapSoftmax()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *first_dims, height, width = inputs.shape
        inputs = inputs.view(*first_dims, width * height)

        quantized_inputs = quantize(
            inputs, total_bits=self.total_bits, frac_bits=self.frac_bits
        )
        predictions = self.hardware_model(quantized_inputs)
        prediction_map = predictions.view(
            *first_dims, self.grid_height, self.grid_width
        )

        return self.softmax(prediction_map)
