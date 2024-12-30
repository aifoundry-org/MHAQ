import torch

from src.aux.types import QScheme
from src.loggers.default_logger import logger

from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct
#from src.quantization.rniq.rniq import Quantizer


class ModelStats:
    def __init__(self, model: torch.nn.Module):
        self.named_params = {name: p for name,
                             p in model.cpu().named_parameters()}
        self.modules = [(name, m) for name, m in model.cpu().named_modules()]
        self.noisy_layers = [
            m for _, m in self.modules if isinstance(m, (NoisyLinear, NoisyConv2d))
        ]

    def _filter_named_params(self, key):
        return {
            name: p.data.item() for name, p in self.named_params.items() if key in name
        }

    def _get_activation_params(self, param_type):
        return self._filter_named_params(f"act_{param_type}")

    def _get_s_activations(self):
        return self._get_activation_params("s")

    def _get_q_activations(self):
        return self._get_activation_params("q")

    def _get_b_activations(self):
        return self._get_activation_params("b")

    def _get_s_weights(self):
        return self._filter_named_params("wght_s")

    def _get_weights_stats(self):
        def condition(name):
            return "conv2d.weight" in name or "lin.weight" in name

        param_values = {
            name: p.abs() for name, p in self.named_params.items() if condition(name)
        }
        stats = {}
        for stat_name, stat_func in [
            ("mean", torch.mean),
            ("std", torch.std),
            ("min", torch.min),
            ("max", torch.max),
        ]:
            stats[stat_name] = {name: stat_func(
                p) for name, p in param_values.items()}
        return stats

    def _compute_module_stats(self, module_condition):
        stat_funcs = [
            torch.mean,
            torch.std,
            torch.min,
            torch.max,
            lambda x: len(x.unique()),
        ]
        stats = [[] for _ in range(5)]

        for name, module in self.modules:
            if module_condition(module):
                weight = (
                    module.weight
                    if module.bias is None
                    else torch.cat((module.weight.ravel(), module.bias))
                )
                for i, func in enumerate(stat_funcs):
                    stats[i].append((name, func(weight.abs()).item()))

        return stats

    def _get_module_weight_stats(self):
        return self._compute_module_stats(
            lambda m: isinstance(m, (NoisyLinear, NoisyConv2d))
        )
    

def get_layer_weights_bit_width(
        layer_weights: torch.Tensor, log_s: torch.Tensor, config=QScheme.PER_TENSOR):
    if config == QScheme.PER_TENSOR:
        min = layer_weights.amin()
        max = layer_weights.amax()
    elif config == QScheme.PER_CHANNEL:
        min = layer_weights.amin((1, 2, 3))
        max = layer_weights.amax((1, 2, 3))

    # add 1 lsb gap to prevent overflow
    log_q = torch.log2((max - min).reshape(log_s.shape) + torch.exp2(log_s))        

    return get_activations_bit_width(log_q, log_s, 0)


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


def get_weights_bit_width_mean(model: torch.nn.Module):
    lin_layers = [
        m for m in model.modules() if isinstance(m, (NoisyConv2d, NoisyLinear))
    ]
    bit_widths = []
    for module in lin_layers:
        weight = module.weight.detach()

        # we are not quantizing bias therefore noo need to include it
        # weight = (
        # module.weight.detach()
        # if module.bias is None
        # else torch.cat((module.weight.detach().reshape(-1), module.bias.detach()))
        # )
        layer_bw = get_layer_weights_bit_width(
            weight, module.log_wght_s.detach(), module.qscheme
        )
        if not torch.isnan(layer_bw):
            bit_widths.append(layer_bw.mean())
    return torch.stack(bit_widths).mean()


def get_activations_bit_width(log_q, log_s, b):
    #s = torch.pow(2, log_s.ravel())
    #q = torch.pow(2, log_q.ravel())
    #zero_point = torch.zeros(1).to(s.device)
    #ql, qm = b - q / 2, b + q / 2
    #Q = Quantizer(s, zero_point, ql, qm)
    #Q.rnoise_ratio = torch.tensor([0]).to(s.device)
    #return torch.ceil(torch.log2(Q.quantize(qm) - Q.quantize(ql) + 1)).mean()
    return (log_q - log_s).mean()
