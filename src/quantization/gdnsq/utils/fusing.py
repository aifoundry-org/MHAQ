import torch

from torch import nn
from operator import attrgetter
from src.aux.qutils import attrsetter


def fuse_conv_bn(model: nn.Module, conv_name: str, bn_name: str):
        conv = attrgetter(conv_name)(model)

        W = conv.weight.clone()
        if conv.bias is not None:
            b = conv.bias.clone()
        else:
            b = torch.zeros(conv.out_channels, device=W.device)

        bn = attrgetter(bn_name)(model)
        mu = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        gamma = bn.weight
        beta = bn.bias

        std = torch.sqrt(var + eps)
        scale = gamma / std
        shape = [-1] + [1] * (W.dim() - 1)

        conv.weight.data = W * scale.view(shape)
        conv.bias = nn.Parameter(beta + (b - mu) * scale)

        attrsetter(bn_name)(model, nn.Identity())  # Replacing bn module with Identity