import torch.nn as nn

from src.loggers.default_logger import logger
from src.quantization.gdnsk.layers.gdnsk_act import NoisyAct

def forward_hook_register(module: nn.Module, hook):
    for c in module.named_children():
        module.get_submodule(c[0]).register_forward_hook(hook(c[0]))
    logger.debug(f"Hook '{hook.name}' registered!")

def pre_forward_hook_register(module: nn.Module, hook):
    for c in module.named_children():
        module.get_submodule(c[0]).register_forward_pre_hook(hook(c[0]))
    logger.debug(f"Hook '{hook.name}' registered!")

def register_lightning_activation_forward_hook(module, hook):
    handlers = []
    for name, m in module.named_modules():
        if isinstance(m, NoisyAct):
            handlers.append(m.register_forward_hook(hook=hook))
            logger.debug(f"Hook for layer {name} of type {type(m)} have registered!")
    return handlers

