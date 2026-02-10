import torch
import numpy as np

from typing import Tuple
from src.aux.types import QScheme
from src.loggers.default_logger import logger

from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.gdnsq_act import NoisyAct

class ModelStats:
    def __init__(self, supported_layers: Tuple):
        self.supported_layers = supported_layers
        self.true_weights_width = None
        self.layer_bit_width_mean = None
        self.activations_bit_width_mean = None
    
    def update_model_state(self, model: torch.nn.Module):
        # updating cached values
        self.true_weights_width = None 
        self.layer_bit_width_mean = None
        self.activations_bit_width_mean = None

        self.named_params = {name: p for name,
                             p in model.named_parameters()}
        self.modules = [(name, m) for name, m in model.named_modules()]
        self.noisy_layers = [
            m for _, m in self.modules if isinstance(m, self.supported_layers)
        ]
        self.noisy_acts = [module for module in model.modules() if isinstance(module, NoisyAct)]
        self.model = model
    
    def get_true_weights_width(self, max=True):
        if not self.true_weights_width:
            bit_widths = []
            for layer in self.noisy_layers:
                layer_bw = get_true_layer_bit_width(layer)
                bit_widths.append(layer_bw)
            
            self.true_weights_width = np.max(bit_widths) if max else np.mean(bit_widths)

        return self.true_weights_width
    
    def get_weights_bit_width_mean(self):
        if not self.layer_bit_width_mean:
            bit_widths = []
            for layer in self.noisy_layers:
                weight = layer.weight.detach()

                layer_bw = get_layer_wnb_bit_width(
                    weight,
                    layer.log_wght_s.detach(),
                    layer.qscheme
                )

                if not torch.isnan(layer_bw):
                    bit_widths.append(layer_bw.mean())
            
            self.layer_bit_width_mean = torch.stack(bit_widths).mean()
        
        return self.layer_bit_width_mean
    
    def get_activations_bit_width_mean(self):
        if not self.activations_bit_width_mean:
            self.activations_bit_width_mean = torch.stack(
                [
                    get_activations_bit_width(
                        module.log_act_q.detach(),
                        module.log_act_s.detach(),
                        module.act_b.detach(),
                    )
                    for module in self.noisy_acts
                ]
            ).mean()
        
        return self.activations_bit_width_mean
    
    # the difference here is that module.bw is updated inside NoiseAct with each forward pass
    def get_true_activations_width(self, model: torch.nn.Module, max=True):
        act_modules = [m for m in model.modules() if isinstance(m, (NoisyAct))]
        bit_widths = []
        for module in act_modules:
            bit_widths.append(module.bw.cpu())
        
        return np.max(bit_widths) if max else np.mean(bit_widths)
    
    def is_converged(self):
        loss = self.model.wrapped_criterion
        converged = (
            self.get_true_weights_width() <= loss.wt
            and get_true_activations_width(self.model) <= loss.at
        )

        return converged



def get_true_layer_bit_width(module: torch.nn.Module, max=True):
    if module.qscheme == QScheme.PER_TENSOR:
        qweights = module.Q.quantize(module.weight.detach())
        bit_width = np.log2(val_count(qweights))
        return bit_width
    elif module.qscheme == QScheme.PER_CHANNEL:
        channel_dim = torch.tensor(0)
        qweights = module.Q.quantize(module.weight.detach())
        # qbiases = module.Q_b.quantize(module.bias.detach())
        reshaped = qweights.permute(
            channel_dim, *
            [i for i in range(qweights.dim()) if i != channel_dim]
        ).reshape(qweights.size(channel_dim), -1)
        bit_widths = [(np.log2(val_count(channel))) for channel in reshaped]
        # bias_bw = np.log2(val_count(qbiases))
        # return (np.max(bit_widths) + np.max(bias_bw)) / 2 if max else (np.mean(bit_widths) + np.mean(bias_bw)) / 2
        return np.max(bit_widths) if max else np.mean(bit_widths)


# much faster than unique
def val_count(q):
    minmax = q.aminmax()
    return (minmax.max - minmax.min + 1).item()


def get_layer_wnb_bit_width(
    layer_weights: torch.Tensor,
    log_s: torch.Tensor,
    # layer_bias: torch.Tensor | None = None,
    # log_b_s: torch.Tensor | None = None,
    config=QScheme.PER_TENSOR,
):
    # add 0.5 bit gap to prevent overflow
    if config == QScheme.PER_TENSOR:
        min = layer_weights.amin()
        max = layer_weights.amax()

        # min_b = layer_bias.amin()
        # max_b = layer_bias.amax()
    elif config == QScheme.PER_CHANNEL:
        # min = layer_weights.amin((1, 2, 3))
        # max = layer_weights.amax((1, 2, 3))
        reduce_dims = tuple(range(1, layer_weights.dim()))
        min = layer_weights.amin(reduce_dims)
        max = layer_weights.amax(reduce_dims)

        # min_b = layer_bias.amin()
        # max_b = layer_bias.amax()

    # add 1 lsb gap to prevent overflow
    log_q = torch.log2((max - min).reshape(log_s.shape) + torch.exp2(log_s))
    # log_q_b = torch.log2(
        # (max_b - min_b).reshape(log_b_s.shape) + torch.exp2(log_b_s))

    # return (get_activations_bit_width(log_q, log_s, 0) + get_activations_bit_width(log_q_b, log_b_s, 0)) / 2
    return (get_activations_bit_width(log_q, log_s, 0))


def get_activations_bit_width_mean(model: torch.nn.Module):
    noisy_layers = [
        module for module in model.modules() if isinstance(module, NoisyAct)
    ]
    return torch.stack(
        [
            get_activations_bit_width(
                module.log_act_q.detach(),
                module.log_act_s.detach(),
                module.act_b.detach(),
            )
            for module in noisy_layers
        ]
    ).mean()


def get_true_weights_width(model: torch.nn.Module, max=True):
    lin_layers = [
        m for m in model.modules() if isinstance(m, (NoisyConv2d, NoisyLinear))
    ]
    bit_widths = []
    for module in lin_layers:
        layer_bw = get_true_layer_bit_width(module)
        bit_widths.append(layer_bw)

    return np.max(bit_widths) if max else np.mean(bit_widths)


# it's a hack to store activations bit widths inside NoisyAct module
# in this function we just collect them
def get_true_activations_width(model: torch.nn.Module, max=True):
    act_modules = [m for m in model.modules() if isinstance(m, (NoisyAct))]
    bit_widths = []
    for module in act_modules:
        bit_widths.append(module.bw.cpu())
    #         bit_widths.append(module.bw.numpy())

    return np.max(bit_widths) if max else np.mean(bit_widths)


def get_weights_bit_width_mean(model: torch.nn.Module):
    lin_layers = [
        m for m in model.modules() if isinstance(m, (NoisyConv2d, NoisyLinear))
    ]
    bit_widths = []
    for module in lin_layers:
        weight = module.weight.detach()
        # if module.bias is not None:
        #     bias = module.bias.detach()
        # else:
        #     bias = torch.Tensor([0]).to(weight.device)

        # we are not quantizing bias therefore noo need to include it
        # weight = (
        # module.weight.detach()
        # if module.bias is None
        # else torch.cat((module.weight.detach().reshape(-1), module.bias.detach()))
        # )
        # layer_bw = get_layer_wnb_bit_width(
            # weight, module.log_wght_s.detach(), module.qscheme
        # )
        layer_bw = get_layer_wnb_bit_width(
            weight, 
            module.log_wght_s.detach(), 
            # bias, 
            # module.log_b_s.detach(), 
            module.qscheme
        )

        if not torch.isnan(layer_bw):
            bit_widths.append(layer_bw.mean())
    return torch.stack(bit_widths).mean()


def get_activations_bit_width(log_q, log_s, b):
    # s = torch.pow(2, log_s.ravel())
    # q = torch.pow(2, log_q.ravel())
    # zero_point = torch.zeros(1).to(s.device)
    # ql, qm = b - q / 2, b + q / 2
    # Q = Quantizer(s, zero_point, ql, qm)
    # Q.rnoise_ratio = torch.tensor([0]).to(s.device)
    # return torch.ceil(torch.log2(Q.quantize(qm) - Q.quantize(ql) + 1)).mean()
    return (log_q - log_s).mean()


def is_converged(model):
    loss = model.wrapped_criterion
    converged = (
        get_true_weights_width(model) <= loss.wt
        and get_true_activations_width(model) <= loss.at
    )
    return converged
