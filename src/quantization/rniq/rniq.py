import torch

from torch import Tensor
from torch.autograd import Function

class QNoise(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, scale):
        output = scale * (torch.round(input) - input)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, scale = inputs
        ctx.save_for_backward(input, scale)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, scale = ctx.saved_tensors
        grad_input = grad_scale = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 0
        if ctx.needs_input_grad[1]:
            #grad_scale = grad_output * (torch.round(input) - input)
            #grad_scale = grad_output * (torch.randint(2, size=input.shape, dtype=input.dtype, device=input.device).sub(0.5))
            grad_scale = grad_output * torch.normal(0, 0.2888, size=input.shape, dtype=input.dtype, device=input.device)
            #grad_scale = grad_output * (torch.rand_like(input).sub_(0.5))

        return grad_input, grad_scale
    
def scaled_noise(x, s):
    return QNoise.apply(x, s)

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

            #qnoise = self._get_qnoise(value, self.scale)
            rnoise = self._get_rnoise(value, self.scale)

            #noise = self.rnoise_ratio * rnoise + (1 - self.rnoise_ratio) * qnoise
            noise = rnoise
            
            value = value + noise / self.scale

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

    def _get_qnoise(self, value: Tensor, scale: Tensor):
        return (torch.round(value) - value).detach() * scale

    def _get_rnoise(self, value: Tensor, scale: Tensor):
        #return torch.randint(2, size=value.shape, dtype=value.dtype, device=value.device).sub(0.5).detach()
        return scaled_noise(value, scale)
