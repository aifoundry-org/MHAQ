import torch

from torch import Tensor
from torch.nn.modules import Module 


class Quantizer:
    def __init__(
        self,
        module: torch.nn.modules.Module,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        rnoise_ratio: torch.Tensor=torch.Tensor([-1.0,])
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
        self.module = module
        self.scale = scale
        self.zero_point = zero_point  # zero point
        self.min_val = min_val
        self.max_val = max_val
        self.rnoise_ratio = torch.Tensor([rnoise_ratio])

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

        # clamp is used only for activations
        # the clamp is before noise beacause adding rounding noise is equivalent to rounding clamp
        value = torch.clamp(value, min=self.min_val, max=self.max_val)

        value = value - self.zero_point

        if self._is_positive_scale():
            value = value / self.scale


        if self.rnoise_ratio.item() == -1.0 or not self._is_positive_scale():
            # No need to calculate noise at all
            rnoise = zero_noise
            qnoise = zero_noise
        elif self.rnoise_ratio.item() == 0.0:
            # Disable random noise calculation
            rnoise = zero_noise
            qnoise = self._get_qnoise(value)
        elif self.rnoise_ratio.item() == 1.0:
            # Disable quantization noise calculation
            qnoise = zero_noise
            rnoise = self._get_rnoise(value)
        else:
            qnoise = self._get_qnoise(value)
            rnoise = self._get_rnoise(value)

        noise = self.rnoise_ratio * rnoise + (1 - self.rnoise_ratio) * qnoise
        
        value = value + noise.detach()

        #assert valid values
        if not self.module.training:
            if self._is_positive_scale() and (torch.any(value < (self.min_val - self.zero_point) / self.scale - 0.5)):
                raise AssertionError("Not all elements in the tensor above min val")
            if self._is_positive_scale() and (torch.any(value > (self.max_val - self.zero_point) / self.scale + 0.5)):
                raise AssertionError("Not all elements in the tensor below max val")            
            if self.rnoise_ratio == 0 and not torch.all(value == value.floor()):
                raise AssertionError("Not all elements in the tensor have integer values.")
        
        return value

    def dequantize(self, quantized_value):
        """
        Dequantizes the input value and
        adds the bias back.
        """
        if self._is_positive_scale():
            return quantized_value * self.scale + self.zero_point

        return quantized_value + self.zero_point

    def _get_qnoise(self, value: Tensor):
        return torch.round(value) - value

    def _get_rnoise(self, value: Tensor):
        return torch.randint(2, size=value.shape, dtype=value.dtype, device=value.device).sub(0.5)
