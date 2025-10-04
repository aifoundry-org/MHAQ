import torch

from torch import Tensor
from torch.autograd import Function
import torch.distributed as dist

class QNoise(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, scale):
        output = torch.round(input) - input
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
            # STE
            #grad_input = grad_output * 0

            # EWGS
            # e = torch.round(input) - input
            # extra gradient so that total grad to x becomes
            # g + delta * |g| * (x - round(x))  (i.e., EWGS Eq. (4))
            # delta = 1e-2
            # grad_input = -torch.abs(grad_output) * e * delta

            # AEWGS
            e = torch.round(input) - input

            num_full = grad_output.sign() * e
            den_full = e.square()

            num = reduce_to_shape(num_full, scale).detach()
            den = reduce_to_shape(den_full, scale).detach()

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(num, op=dist.ReduceOp.AVG)
                dist.all_reduce(den, op=dist.ReduceOp.AVG)

            delta = num / (den + 1e-6)

            # prevent gradient vanish
            g_scale = (delta * num_full).clamp_max(0.99) 
            
            grad_input = -grad_output * g_scale
            

        if ctx.needs_input_grad[1]:
            # correct scaling accoring to https://arxiv.org/abs/2508.14004
            grad_scale = (3.0 ** -0.5) * grad_output * (torch.randint(2, size=input.shape, dtype=input.dtype, device=input.device).sub(0.5))            

        return grad_input, grad_scale
    


def reduce_to_shape(t: Tensor, like: Tensor) -> Tensor:
    dims_to_reduce = [i for i, size in enumerate(like.shape) if size == 1]
    return  torch.mean(t, dim=tuple(dims_to_reduce), keepdim=True)

    
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
        self.positive_scale = torch.all(torch.as_tensor(self.scale) > 0).item()


    def quantize(self, value):
        """
        Quantizes the input value after
        clamping it to the specified range.
        """

        # clamp is used only for activations
        # the clamp is before noise beacause adding rounding noise is equivalent to rounding clamp
        value = torch.clamp(value, min=self.min_val, max=self.max_val)

        value = value - self.zero_point

        if not self.positive_scale:
            return value
            
        value = value / self.scale

        noise = self._get_rnoise(value, self.scale)
         
        value = value + noise

        #assert valid values
        if not self.module.training:
            if torch.any(value < torch.floor((self.min_val - self.zero_point) / self.scale)):
                raise AssertionError("Not all elements in the tensor above min val")
            if torch.any(value > torch.ceil((self.max_val - self.zero_point) / self.scale)):
                raise AssertionError("Not all elements in the tensor below max val")            
            if not torch.all((value == value.floor()) | (value == value.ceil())):
                raise AssertionError("Not all elements in the tensor have integer values.")
        
        return value

    def dequantize(self, quantized_value):
        """
        Dequantizes the input value and
        adds the bias back.
        """        
        if not self.positive_scale:
            return quantized_value + self.zero_point
            
        return quantized_value * self.scale + self.zero_point
        


    def _get_rnoise(self, value: Tensor, scale: Tensor):
        return scaled_noise(value, scale)
