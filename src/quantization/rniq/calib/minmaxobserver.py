import torch

from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct
from src.loggers.default_logger import pl_logger
from src.aux.traverse import previous_leaf
import numpy as np


class ObserverHook:
    def __init__(self) -> None:
        pass

    def __call__(self, layer_name=None, *args):
        raise NotImplementedError("You need to implement __call__ method!")


class MinMaxObserver(ObserverHook):
    def __init__(self) -> None:
        super().__init__()
        self._min_values = torch.tensor([], device="cuda")
        self._max_values = torch.tensor([], device="cuda")

    def __call__(self, module, input, output):
        return self._hook(module, input, output)

    def _hook(self, module, input, output) -> None:
        try:
            module.min_values = torch.cat(
                (module.min_values, torch.min(input[0]).reshape(1)))
            module.max_values = torch.cat(
                (module.max_values, torch.max(input[0]).reshape(1)))
        except AttributeError:
            module.min_values = torch.min(input[0]).reshape(1)
            module.max_values = torch.max(input[0]).reshape(1)


def apply_mean_stats_activations(module, abits=8, max_bits = 24):    
    for name, m in module.named_modules():

        if isinstance(m, NoisyAct):
            min = m.min_values.min()
            max = m.max_values.max()

            pl_logger.info(f"Min: {min}, Max: {max}, prev_leaf: {previous_leaf(module, name)}")

            m.min_values = torch.Tensor([])
            m.max_values = torch.Tensor([])

            if not m.act_b.requires_grad:
                # keep offset
                min = m.act_b.data.to(min)

            if not m.log_act_q.requires_grad and not m.log_act_s.requires_grad:
                abits = max_bits

            if max - min > 0:
                # not zero width
                log_s = torch.log2((max - min) / (2**abits - 1))
                log_q = log_s + abits

                m.act_b = torch.nn.Parameter(torch.tensor([min]), requires_grad=m.act_b.requires_grad)
                m.log_act_q = torch.nn.Parameter(torch.tensor([log_q]), requires_grad=m.log_act_q.requires_grad)
                m.log_act_s = torch.nn.Parameter(torch.tensor([log_s]), requires_grad=m.log_act_s.requires_grad)
            else:
                # pruned 
                m.log_act_q = torch.nn.Parameter(torch.tensor([0]), requires_grad=False)
                m.log_act_s = torch.nn.Parameter(torch.tensor([0]), requires_grad=False)
                m.act_b = torch.nn.Parameter(torch.tensor([min]), requires_grad=False)


def apply_quantile_weights_s(module, wbits=8, max_bits = 24, qscheme="per-channel"):

    for name, m in module.named_modules():
        # TODO: qscheme
        if isinstance(m, (NoisyLinear, NoisyConv2d)):
            max = m.weight.data.amax((1, 2, 3))
            min = m.weight.data.amin((1, 2, 3))

            if not m.log_wght_s.requires_grad:
                wbits = max_bits

            # XXX: handle max-min == 0
            log_s = torch.max(torch.tensor(m.log_wght_s), torch.log2((max - min) / (2**wbits - 1)).reshape(m.log_wght_s.shape))

            #m.log_wght_s = torch.nn.Parameter(torch.full(m.log_wght_s.shape, log_s), requires_grad=m.log_wght_s.requires_grad) # per-tensor
            m.log_wght_s = torch.nn.Parameter(log_s, requires_grad=m.log_wght_s.requires_grad)

            pl_logger.debug(f"{name}_q = {m.log_wght_s} ")
