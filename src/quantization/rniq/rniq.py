import torch

from torch import Tensor
from src.quantization.rniq.utils.enums import QMode


class Quantizer:
    def __init__(
        self,
        scale: float,
        zero_point: float,
        min_val: float,
        max_val: float,
        rnoise_ratio: float = -1.0,
    ) -> None:
        """
        Main quantizer for rniq method.

        Args:
            scale (float): _description_
            zero_point (float): _description_
            min_val (float): _description_
            max_val (float): _description_
            rnoise_ratio (float): _description_
        """
        self.scale = scale
        self.zero_point = zero_point  # zero point
        self.min_val = min_val
        self.max_val = max_val
        self.rnoise_ratio = rnoise_ratio

    def _is_positive_scale(self):
        """
        Check if the scale is positive
        for both float and tensor types.
        """
        if isinstance(self.scale, float):
            return self.scale > 0
        elif isinstance(self.scale, torch.Tensor):
            return torch.all(self.scale > 0)
        return False

    def quantize(self, value):
        """
        Quantizes the input value after
        clamping it to the specified range.
        """

        # This conditions are not essential
        # Just for sake of opitmization

        zero_noise = torch.zeros_like(value)

        if self.rnoise_ratio == -1.0 or not self._is_positive_scale():
            # Disable all noise calculation
            qnoise = rnoise = zero_noise
        elif self.rnoise_ratio == 0.0:
            # Disable random noise calculation
            rnoise = zero_noise
            qnoise = self._get_qnoise(value)
        elif self.rnoise_ratio == 1.0:
            # Disable quantization noise calculation
            qnoise = zero_noise
            rnoise = self._get_rnoise(value)
        else:
            qnoise = self._get_qnoise(value)
            rnoise = self._get_rnoise(value)

        noise = self.scale * (
            self.rnoise_ratio * rnoise + (1 - self.rnoise_ratio) * qnoise
        )

        clamped_value = (
            # torch.clamp(value / self.scale, min=self.min_val, max=self.max_val) - self.zero_point
            torch.clamp(value, min=self.min_val, max=self.max_val)
        )

        # if self._is_positive_scale():
        # return torch.floor(clamped_value / self.scale + 0.5)

        return clamped_value + noise.detach()

    def dequantize(self, quantized_value):
        """
        Dequantizes the input value and
        adds the bias back.
        """
        if self._is_positive_scale():
            return quantized_value * self.scale + self.zero_point

        return quantized_value + self.zero_point

    def _get_qnoise(self, value: Tensor):
        return torch.clamp(
            torch.round(value / self.scale - self.zero_point),
            min=self.min_val,
            max=self.max_val,
        ) - value

    def _get_rnoise(self, value: Tensor):
        return torch.randint(low=-1, high=0, size=value.shape, dtype=value.dtype).add(
            0.5
        )
