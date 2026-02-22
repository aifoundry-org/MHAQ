import torch

from torch import nn

from src.aux.types import QScheme
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.quantization.gdnsq.layers.gdnsq_act import NoisyAct
from src.aux.qutils import is_biased

def iso_fro_loss(W: torch.Tensor, eps: float = 1e-8):
    C_out = W.shape[0]
    W2 = W.view(C_out, -1)
    scale = (W2.norm(p='fro')**2 / max(C_out,1)).clamp_min(eps)
    G = (W2 @ W2.t())
    I = scale * torch.eye(C_out, device=W.device, dtype=W.dtype)
    return ((G - I)**2).sum()

class ModelHelper:
    @staticmethod
    def get_model_values(model: nn.Module, qscheme: QScheme = QScheme.PER_TENSOR):
        log_b_s, log_wght_s, log_w_n_b, log_act_q, log_act_s, balance = [], [], [], [], [], []


        # Helper to handle log_s and log_w_n_b collection
        def collect_log_weights(module):
            if module.log_wght_s.requires_grad:
                # add 0.5 bit gap to prevent overflow
                if qscheme == QScheme.PER_CHANNEL:
                    log_wght_s.append(module.log_wght_s.ravel())
                    # log_wght_s.append(module.log_b_s)
                    min = module.weight.amin((1,2,3))
                    max = module.weight.amax((1,2,3))

                    # if is_biased(module):
                    #     min_b = module.bias.amin()
                    #     max_b = module.bias.amax()
                    # else:
                    #     min_b = torch.Tensor([0]).to(min.device)
                    #     max_b = torch.Tensor([0]).to(max.device)
                    bal = iso_fro_loss(module.weight)
                    balance.append(bal.ravel())

                elif qscheme == QScheme.PER_TENSOR:
                    log_wght_s.append(module.log_wght_s)
                    # log_wght_s.append(module.log_b_s)
                    min = module.weight.amin()
                    max = module.weight.amax()

                    # min_b = module.bias.amin()
                    # max_b = module.bias.amax()

                    bal = iso_fro_loss(module.weight).mean()                    
                    balance.append(bal)

                # add 1 lsb gap to prevent overflow
                log_w_n_b.append(torch.log2(max - min + torch.exp2(module.log_wght_s.ravel())))
                # log_w_n_b.append(torch.log2(max_b - min_b + torch.exp2(module.log_b_s)))
                    

        # Helper to handle log_act_q and log_act_s collection
        def collect_log_activations(module):
            if module.log_act_s.requires_grad:
                log_act_q.append(module.log_act_q)
                log_act_s.append(module.log_act_s)

        for name, module in model.named_modules():
            if isinstance(module, (NoisyConv2d, NoisyLinear)): # TODO watch supported layers!
                collect_log_weights(module)
            elif isinstance(module, NoisyAct):
                collect_log_activations(module)

        # Stack or concatenate the results based on the quantization scheme
        if qscheme == QScheme.PER_TENSOR:
            res = (
                torch.stack(log_act_s).ravel(),
                torch.stack(log_act_q).ravel(),
                torch.stack(log_wght_s).ravel(),
                torch.stack(log_w_n_b).ravel(),
                torch.stack(balance).ravel()
            )
        elif qscheme == QScheme.PER_CHANNEL:
            res = (
                torch.cat(log_act_s),
                torch.cat(log_act_q),
                torch.cat(log_wght_s),
                torch.cat(log_w_n_b),
                torch.cat(balance)
            )

        return res
