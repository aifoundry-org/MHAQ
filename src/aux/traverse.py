import torch.nn as nn

def previous_leaf(model: nn.Module, target_name: str):
    """
    Return (prev_name, prev_module) for the leaf module that appears immediately
    before `target_name` in depth-first preorder traversal of `model`.
    Root (“”) and container modules (those that have children) are skipped.

    If `target_name` is the first leaf, returns (None, None).
    Raises KeyError if `target_name` is not a leaf or not found.
    """
    prev = (None, None)
    for name, mod in model.named_modules():
        if name == '' or any(mod.children()):  # skip root & containers
            continue
        if name == target_name:
            return prev
        prev = (name, mod)
    raise KeyError(f"{target_name!r} is not a leaf module in this model")