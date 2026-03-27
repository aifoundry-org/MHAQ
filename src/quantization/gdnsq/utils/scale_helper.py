import torch
from torch import Tensor
from src.quantization.gdnsq.gdnsq_utils import GradNoiseType

def compute_grad_scale(grad_noise_type: GradNoiseType, input: Tensor, grad_output: Tensor) -> Tensor:
    if grad_noise_type == GradNoiseType.BER3:
        # Bernoulli ±0.5, variance-corrected: https://arxiv.org/abs/2508.14004
        r = torch.randint_like(input, 2).sub_(0.5)
        return (3.0 ** -0.5) * grad_output * r
    elif grad_noise_type == GradNoiseType.BER1:
        r = torch.randint_like(input, 2).sub_(0.5)
        return grad_output * r
    elif grad_noise_type == GradNoiseType.NORM3:
        noise = torch.randn_like(input)
        return (3.0 ** -0.5) * grad_output * noise * 0.5
    elif grad_noise_type == GradNoiseType.NORM1:
        noise = torch.randn_like(input)
        return grad_output * noise * 0.5
    elif grad_noise_type == GradNoiseType.UNIFORM:
        noise = torch.rand_like(input).sub_(0.5)
        return grad_output * noise
    elif grad_noise_type == GradNoiseType.ROUNDING:
        e = torch.round(input) - input
        return grad_output * e
    else:
        raise AttributeError(f"Unknown grad_noise_type: {grad_noise_type}")
