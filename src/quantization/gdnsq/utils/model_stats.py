import torch
import numpy as np

from src.aux.types import QScheme
from src.loggers.default_logger import logger

from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.gdnsq_act import NoisyAct

# from src.quantization.gdnsq.gdnsq import Quantizer


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
            val_count,
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

    def print_stats(self):
        weights_stats = self._get_module_weight_stats
        sections = [
            ("Model S activations", self._get_s_activations()),
            ("Model Q activations", self._get_q_activations()),
            ("Model B activations", self._get_b_activations()),
            ("Model S weights", self._get_s_weights()),
            (
                "Model weights abs mean, std",
                zip(weights_stats()[0], weights_stats()[1]),
            ),
            ("Model weights abs min, max", zip(
                weights_stats()[2], weights_stats()[3])),
            (
                "Model weights bit_width",
                [
                    (i[0], get_activations_bit_width(
                        torch.log2(i[1]) + 1, j[1], 0))
                    for i, j in zip(weights_stats()[3], self._get_s_weights())
                ],
            ),
        ]
        for title, values in sections:
            logger.debug(f"\n{title}")
            for name, value in values:
                logger.debug(f"{name}: {value}")


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
