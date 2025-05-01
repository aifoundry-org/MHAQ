import pickle
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from pytorchcv.model_provider import get_model


def replace_relu_with_softplus(module: nn.Module, beta: float = 1.0, threshold: float = 20.0):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.Softplus(beta=beta, threshold=threshold))
        else:
            replace_relu_with_softplus(child, beta=beta, threshold=threshold)


def normalize_batch_to_unit_range(batch: torch.Tensor) -> torch.Tensor:
    mins = batch.quantile(0.01)
    maxs = batch.quantile(0.99)
    denom = (maxs - mins).clamp(min=1e-6)
    return ((batch - mins) / denom).clamp(0, 1).mul(255).round()


def tv_loss(f):
    # f: (B, C, H, W)
    dh = f[:, :, 1:, :]   - f[:, :, :-1, :]   # shape (B,C,H-1,W)
    dw = f[:, :, :, 1:]   - f[:, :, :, :-1]   # shape (B,C,H,W-1)
    grad2 = dh.abs().sum() + dw.abs().sum()  # (B, H-1, W-1)
    # isotropic vector-TV
    return grad2


def tv_feature_loss(f):
    # f: (B, C, H, W)
    # whiten
    f = (f - f.mean(dim=[0,2,3], keepdim=True)) / (f.std(dim=[0,2,3], keepdim=True) + 1e-5)
    # compute spatial diffs
    dh = f[:, :, 1:, :-1] - f[:, :, :-1, :-1]  # -> (B, C, H-1, W-1)
    dw = f[:, :, :-1, 1:] - f[:, :, :-1, :-1]  # -> (B, C, H-1, W-1)
    # squared magnitude of the C-dimensional gradient vector
    grad2 = dh.abs().sum() + dw.abs().sum()  # (B, H-1, W-1)
    # isotropic vector-TV
    return grad2


def main():

    g = torch.Generator()
    g.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #class_id = args.class_id
    batch_size = 484
    num_steps = 1000
    sigma = 1
    lr = 0.02
    beta = 10.
    l_inv = 1e-7  # deepinverse regularization weight
    l_tv = 1e-5  # image total variation regularization weight
    l_tv_f = 2e-9  # feature total variation regularization weight
    # output_dir = 'output/test'
    output_dir = 'output/train'
    file_format = "pkl"
    prefix = "data_batch"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and prepare model
    model = get_model("resnet20_cifar100", pretrained=True)
    replace_relu_with_softplus(model, beta=beta)
    model = model.to(device).eval().requires_grad_(False)

    # 1b. Register hooks: target, features, and all BatchNorm outputs
    activation = {}
    hooks = {}
    # BatchNorm hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            def bn_hook_fn(module, inp, out, key):
                # assume bn_inputs[name] holds the pre‐norm tensor x for that layer
                mean_b = inp[0].mean(dim=[0,2,3])            # (C,)
                var_b  = inp[0].var(dim=[0,2,3], unbiased=False)  # (C,)

                running_mean = module.running_mean                   # (C,)
                running_var  = module.running_var                    # (C,)

                # DeepInversion BN regularizer for that layer:
                reg = ((mean_b - running_mean)**2).sum() + ((var_b  - running_var)**2).sum()
                reg2 = tv_feature_loss(out)

                activation.__setitem__(key, (reg, reg2))


            key = f'bn_{name}'
            hooks[key] = module.register_forward_hook(
                lambda module, inp, out, key=key: bn_hook_fn(module, inp, out, key))

    # 2. Load CIFAR-100 subset
    transform = transforms.ToTensor()
    cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    # cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    #indices = [i for i, (_, y) in enumerate(cifar_raw) if y == class_id]
    #subset = Subset(cifar_raw, indices)

    # loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=True, generator=g)
    loader = DataLoader(cifar100, batch_size=batch_size, shuffle=True, generator=g)
    
    
    # 3. Normalize for input
    # mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1,3,1,1)
    # std  = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1,3,1,1)

    batch_paths = []

    for batch_idx in trange(len(loader)):
        img_raw_batch, labels = next(iter(loader))

        # !labels should be random for magic to work
        targets = labels.to(device)
        #target = F.softmax(model(img_norm), dim=1)

        # 5. Initialize reconstruction
        x0 = torch.randn_like(img_raw_batch).to(device)
        x = (sigma * x0).requires_grad_(True)
        optimizer = optim.Adam([x], lr=lr)

        activation.clear()

        # 6. Inversion loop
        for step in trange(num_steps + 1):
            optimizer.zero_grad()

            out = model(x)
            # Gather batchnorm outputs and compute deepinverse regularizer
            bn_reg = torch.tensor(0.0, device=device)
            tv_f_reg = torch.tensor(0.0, device=device)
            for key, feat in activation.items():
                bn_reg += feat[0]
                tv_f_reg += feat[1]

            loss = F.cross_entropy(out, targets)
            #loss  = F.kl_div(F.log_softmax(out, dim=1) , target, reduction='batchmean')
            tv_reg = tv_loss(x)
            total_loss = loss + l_tv * tv_reg + l_tv_f * tv_f_reg  + l_inv * torch.sqrt(bn_reg)

            total_loss.backward(retain_graph=False)
            optimizer.step() 

            if step % 50 == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}, BN-Reg: {bn_reg.item():.4f}, TV-Reg: {tv_reg.item():.4f}, TVF-Reg: {tv_f_reg.item():.4f}")
                # x_denorm = normalize_batch_to_unit_range(x)
                # utils.save_image(x_denorm, f'{output_dir}/batch_iter_{step:04d}.png', nrow=int(np.sqrt(batch_size)))
            

        x_denorm = normalize_batch_to_unit_range(x)
        
        x_np = x_denorm.detach().cpu().numpy()
        y_np = targets.detach().cpu().numpy().tolist()

        batch_dict = {'data': x_np, 'labels': y_np}
        fname = f"{prefix}_{batch_idx}.{file_format}"
        path = os.path.join(output_dir, fname)
        with open(path, 'wb') as f:
            pickle.dump(batch_dict, f)
        print(f"Saved batch {batch_idx} to {path}")
        batch_paths.append(path)

            

        # 7. Remove hooks
        # for h in hooks.values():
            # h.remove()

        print(f"Optimization complete. Results saved in {output_dir}/")

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
