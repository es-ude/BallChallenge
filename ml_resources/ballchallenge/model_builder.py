from typing import Any, Self

import torch

from elasticai.creator.vhdl.design_creator import DesignCreator
from elasticai.creator.nn import Sequential, Flatten
from elasticai.creator.nn.fixed_point import (
    Conv1d,
    BatchNormedConv1d,
    Linear,
    HardTanh,
    HardSigmoid,
    quantize,
)


class _InputQuantizedSequential(Sequential):
    def __init__(
        self,
        *submodules: DesignCreator,
        total_bits: int,
        frac_bits: int,
    ) -> None:
        super().__init__(*submodules)
        self._total_bits = total_bits
        self._frac_bits = frac_bits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quantized_inputs = quantize(
            x=inputs, total_bits=self._total_bits, frac_bits=self._frac_bits
        )
        return super().forward(quantized_inputs)


class ModelBuilder:
    def __init__(
        self, total_bits: int, frac_bits: int, input_shape: tuple[int, int]
    ) -> None:
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._input_shape = input_shape
        self._submodules: list[DesignCreator] = []

    def build_model(self) -> _InputQuantizedSequential:
        return _InputQuantizedSequential(
            *self._submodules, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    def add_conv1d(self, filters: int, kernel_size: int, bias: bool = True) -> Self:
        return self._add_convolution(Conv1d, filters, kernel_size, bias)

    def add_batch_normed_conv1d(
        self,
        filters: int,
        kernel_size: int,
        bias: bool = True,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bn_affine: bool = True,
    ) -> Self:
        return self._add_convolution(
            BatchNormedConv1d,
            filters=filters,
            kernel_size=kernel_size,
            bias=bias,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            bn_affine=bn_affine,
        )

    def add_linear(self, output_units: int, bias: bool = True) -> Self:
        num_channels, num_features = self._input_shape
        if num_channels is not None:
            raise ValueError(
                f"Cannot add linear layer on 2D input (input shape: {self._input_shape})."
            )
        self._add_submodule(
            Linear,
            total_bits=self._total_bits,
            frac_bits=self._frac_bits,
            in_features=num_features,
            out_features=output_units,
            bias=bias,
        )
        self._input_shape = (None, output_units)
        return self

    def add_flatten(self) -> Self:
        self._add_submodule(Flatten)

        num_channels, signal_length = self._input_shape
        if num_channels is not None:
            self._input_shape = (None, num_channels * signal_length)

        return self

    def add_hard_tanh(self) -> Self:
        return self._add_submodule(
            HardTanh, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    def add_hard_sigmoid(self) -> Self:
        return self._add_submodule(
            HardSigmoid, total_bits=self._total_bits, frac_bits=self._frac_bits
        )

    def _add_submodule(self, module: type[DesignCreator], **module_kwargs: Any) -> Self:
        self._submodules.append(module(**module_kwargs))
        return self

    def _add_convolution(
        self,
        convolution: type[DesignCreator],
        filters: int,
        kernel_size: int,
        bias: bool,
        **conv_kwargs: Any,
    ) -> Self:
        num_channels, signal_length = self._input_shape

        if num_channels is None:
            raise ValueError(
                f"Cannot add convolution for flat input vector (input shape: {self._input_shape})."
            )

        self._add_submodule(
            module=convolution,
            total_bits=self._total_bits,
            frac_bits=self._frac_bits,
            signal_length=signal_length,
            in_channels=num_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            bias=bias,
            **conv_kwargs,
        )
        self._input_shape = (filters, signal_length - (kernel_size - 1))
        return self
