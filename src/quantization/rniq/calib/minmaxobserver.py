import torch

from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct
from src.loggers.default_logger import logger
import numpy as np


class ObserverHook:
    def __init__(self) -> None:
        pass

    def __call__(self, layer_name=None, *args):
        raise NotImplementedError("You need to implement __call__ method!")


class MinMaxObserver(ObserverHook):
    # def __init__(self, q=0.95) -> None:
    def __init__(self, q=0.99) -> None:
        super().__init__()
        self.q = q
        self._min_values = torch.tensor([], device="cuda")
        self._max_values = torch.tensor([], device="cuda")

    def __call__(self, module, input, output):
        return self._hook(module, input, output)

    def _hook(self, module, input, output) -> None:
        def torch_quantile(input, q):
            return torch.quantile(input=input, q=q).reshape(1)

        def np_quantile(input, q):
            if q == 1.:
                res = torch.max(input).reshape(1)
            elif q == 0.:
                res = torch.min(input).reshape(1)
            else:
                cpu_input = input.detach().cpu().numpy()
                np_quantile = np.quantile(cpu_input, q).astype(np.float32)
                res = torch.as_tensor(np_quantile, device=torch.device('cuda')).reshape(1)
            return res

        # def min_quantile(inp):
        #     return -torch.quantile(-torch.min(inp, 0)[0], q=self.q).reshape(1)

        # def max_quantile(inp):
        #     return torch.quantile(torch.max(inp, 0)[0], q=self.q).reshape(1)

        try:
            module.min_values = torch.cat(
                (module.min_values,  np_quantile(input[0], 1-self.q)))
            module.max_values = torch.cat(
                (module.max_values, np_quantile(input[0], self.q)))
        except AttributeError:

            module.min_values = np_quantile(input[0], 1-self.q)
            module.max_values = np_quantile(input[0], self.q)


def apply_mean_stats_activations(module, abits=8):
    for name, m in module.named_modules():
        eps = 1e-2
        mabits = 24

        if isinstance(m, NoisyAct):
            qmin = m.min_values.min().detach()
            qmax = m.max_values.max().detach()

            print((qmin, qmax))

            if qmin < 0:
                qmax = torch.max(qmax, -qmin)
                qmin = -qmax
            else:
                qmin = torch.tensor([0]).to(qmax)

            m.min_values = torch.Tensor([])
            m.max_values = torch.Tensor([])

            if qmax > 0:
                if (m.act_b.requires_grad and m.log_act_q.requires_grad and m.log_act_s.requires_grad):
                    # m.act_b = torch.nn.Parameter((qmin + qmax)/2, requires_grad=True)
                    q = torch.nn.Parameter(torch.tensor(
                        [torch.log2((qmax - qmin) / (1.0 - 1.0 / 2.0**abits))]), requires_grad=True)
                    m.log_act_q = q
                    m.log_act_s = torch.nn.Parameter(
                        torch.tensor([q - abits + eps]), requires_grad=True)
                elif (m.log_act_q.requires_grad and m.log_act_s.requires_grad):
                    m.log_act_q = torch.nn.Parameter(torch.tensor(
                        [torch.log2(qmax / (1.0 - 1.0 / 2.0**abits))]), requires_grad=True)
                    m.log_act_s = torch.nn.Parameter(torch.tensor(
                        [m.log_act_q - abits + eps]), requires_grad=True)
                else:
                    # m.act_b = torch.nn.Parameter(torch.mean(torch.stack((qmin, qmax))), requires_grad=False)
                    q = torch.nn.Parameter(torch.tensor(
                        [torch.log2((qmax - qmin) / (1.0 - 1.0 / 2.0**mabits))]), requires_grad=False)
                    m.log_act_q = q
                    m.log_act_s = torch.nn.Parameter(torch.tensor(
                        [q - mabits + eps]), requires_grad=False)
            else:
                m.log_act_q = torch.nn.Parameter(
                    0.0 * torch.tensor([q]), requires_grad=False)
                m.log_act_s = torch.nn.Parameter(
                    1.0 * torch.tensor([q]), requires_grad=False)


def apply_quantile_weights_s(module, q=0.95, wbits=8, weight_statistic="quantile", qscheme="per-tensor"):

    def max_quantile(module: torch.nn.Module):
        return module.weight.data.ravel().abs().quantile(q)

    for name, m in module.named_modules():
        if isinstance(m, (NoisyLinear, NoisyConv2d)):
            if weight_statistic == "quantile":
                module_max_values = max_quantile(m)
            elif weight_statistic == "max":
                module_max_values = m.weight.data.abs().amax((1, 2, 3))
            else:
                raise NotImplemented(
                    f"statistic '{weight_statistic}' is unknown!")

            # log_q = torch.log2(abs_values + 1)
            # if m.log_wght_s.requires_grad:
                # m.log_wght_s.data = (log_q - torch.log2(torch.exp2((torch.as_tensor(wbits))) - 1)).reshape(m.log_wght_s.shape)
                # m.log_wght_s.data = (log_q - torch.log2(torch.exp2((torch.as_tensor(wbits))) - 1))
            #    m.log_wght_s.data = (
            #        log_q - torch.log2(torch.exp2((torch.as_tensor(wbits))) - 1)).mean()
            # module_max_values = max_quantile(torch.abs(m.weight.data))
            # module_max_values = torch.cat((m.weight.data.ravel(), m.bias.data)).abs().max()
            # module_max_values = m.weight.data.ravel().abs().max()
            # module_max_values = m.weight.data.ravel().abs().quantile(q)

            eps = 1e-2
            mwbits = 24

            if m.log_wght_s.requires_grad:
                log_q = torch.log2(module_max_values /
                                   (1.0 - 1.0 / 2**(wbits - 1)))
                m.weight.data = m.weight.data.clamp(
                    min=-module_max_values, max=module_max_values)
                # m.log_wght_s = torch.nn.Parameter(torch.Tensor([(log_q - wbits + 1 + eps)]), requires_grad=True)
                m.log_wght_s = torch.nn.Parameter(torch.full(
                    m.log_wght_s.shape, log_q - wbits + 1 + eps), requires_grad=True)
            else:
                log_q = torch.log2(module_max_values /
                                   (1.0 - 1.0 / 2**(mwbits - 1)))
                m.log_wght_s = torch.nn.Parameter(torch.full(
                    m.log_wght_s.shape, log_q - mwbits + 1 + eps), requires_grad=False)

            logger.debug(f"{name}_q = {m.log_wght_s} ")
