import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxsim import simplify
from tqdm import tqdm

from src.quantization.gdnsq.layers.gdnsq_act import NoisyAct
from src.quantization.gdnsq.layers.gdnsq_conv2d import NoisyConv2d
from src.quantization.gdnsq.layers.gdnsq_linear import NoisyLinear
from src.aux.types import QScheme


class GDNSQActInference(nn.Module):
    """A lightweight activation layer that uses precomputed constants."""
    def __init__(self, orig_layer: NoisyAct):
        super().__init__()
        self.disable = orig_layer.disable
        if not self.disable:
            s = torch.exp2(orig_layer.log_act_s).detach()
            q = torch.exp2(orig_layer.log_act_q).detach()

            self.register_buffer('scale', s)
            self.register_buffer('zero_point', orig_layer.act_b.detach())
            self.register_buffer('min_val', orig_layer.act_b.detach())
            self.register_buffer(
                'max_val',
                (orig_layer.act_b + q - s).detach()
            )

    def forward(self, x):
        if self.disable:
            return x
        x = torch.clamp(x, min=self.min_val, max=self.max_val)
        x = x - self.zero_point
        x = x / self.scale
        x = torch.round(x)
        x = x * self.scale
        x = x + self.zero_point
        return x


def apply_model_surgery(module: nn.Module):
    """
    Recursively traverses the model and replaces the layers
    with equivalents that have pre-quantized weights.
    """
    for name, child in module.named_children():
        if isinstance(child, NoisyAct):
            setattr(module, name, GDNSQActInference(child))

        elif isinstance(child, NoisyConv2d):
            s = torch.exp2(child.log_wght_s).detach()
            if child.qscheme == QScheme.PER_CHANNEL:
                min_val = child.weight.detach().amin((1, 2, 3), keepdim=True)
            else:
                min_val = child.weight.detach().amin()

            w = child.weight.detach()
            w = w - min_val
            w = w / s
            w = torch.round(w)
            w = w * s + min_val

            b = child.bias.detach() if child.bias is not None else None
            if child.quant_bias and b is not None:
                s_b = s.ravel()
                min_b = min_val.ravel()
                b = b - min_b
                b = b / s_b
                b = torch.round(b)
                b = b * s_b + min_b

            new_conv = nn.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(b is not None),
                padding_mode=child.padding_mode
            )
            new_conv.weight.data = w
            if b is not None:
                new_conv.bias.data = b

            setattr(module, name, new_conv)

        elif isinstance(child, NoisyLinear):
            s = torch.exp2(child.log_wght_s).detach()
            if child.qscheme == QScheme.PER_CHANNEL:
                min_val = child.weight.detach().amin(dim=1, keepdim=True)
            else:
                min_val = child.weight.detach().amin()

            w = child.weight.detach()
            w = w - min_val
            w = w / s
            w = torch.round(w)
            w = w * s + min_val

            b = child.bias.detach() if child.bias is not None else None

            new_linear = nn.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(b is not None)
            )
            new_linear.weight.data = w
            if b is not None:
                new_linear.bias.data = b

            setattr(module, name, new_linear)

        else:
            apply_model_surgery(child)


class ONNXPipeline:
    """A pipeline for exporting, verifying, and validating ONNX models."""
    def __init__(self, qmodel, datamodule, onnx_file_path):
        self.qmodel = qmodel
        self.datamodule = datamodule
        self.onnx_file_path = onnx_file_path

    def export(self, use_simplifier=False):
        self.qmodel.eval()
        self.qmodel.cuda()

        apply_model_surgery(self.qmodel)

        dummy_input = torch.randn(
            1, 3, 32, 32, dtype=torch.float32, device=self.qmodel.device
        )

        torch.onnx.export(
            self.qmodel,
            dummy_input,
            self.onnx_file_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamo=False,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"Successfully exported to {self.onnx_file_path}")

        if use_simplifier:
            model_onnx = onnx.load(self.onnx_file_path)
            model_simp, check = simplify(model_onnx)
            if check:
                onnx.save(model_simp, self.onnx_file_path)
                print(f"Successfully simplified {self.onnx_file_path}")
            else:
                print("Warning: ONNX Simplifier failed to verify graph.")

        return self

    def verify(self):
        """Verify the mathematical parity of the ONNX model."""
        onnx_model = onnx.load(self.onnx_file_path)
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print(f"ONNX model is invalid: {e}")
            return self

        self.qmodel.eval()
        self.qmodel.cuda()

        self.datamodule.setup(stage="validate")
        val_loader = self.datamodule.val_dataloader()
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        images, _ = next(iter(val_loader))
        images = images.cuda()
        batch_size = images.shape[0]

        with torch.no_grad():
            torch_output = self.qmodel(images).cpu().numpy()

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(
            self.onnx_file_path, providers=providers
        )
        input_name = ort_session.get_inputs()[0].name
        ort_output = ort_session.run(
            None, {input_name: images.cpu().numpy()}
        )[0]

        cos_sims = []
        for i in range(batch_size):
            t_flat = torch_output[i].flatten()
            o_flat = ort_output[i].flatten()
            sim = np.dot(t_flat, o_flat) / (
                np.linalg.norm(t_flat) * np.linalg.norm(o_flat)
            )
            cos_sims.append(sim)

        mean_cos_sim = np.mean(cos_sims)

        torch_preds = np.argmax(torch_output, axis=1)
        ort_preds = np.argmax(ort_output, axis=1)

        matched_predictions = np.sum(torch_preds == ort_preds)
        match_rate = (matched_predictions / batch_size) * 100

        print(f"Mean Cosine Similarity: {mean_cos_sim:.4f}")
        print(f"Prediction Match Rate:  {match_rate:.1f}%")

        if match_rate == 100.0:
            print("Verification Passed: 100% logic preservation on real data.")
        elif match_rate >= 95.0:
            print("Verification Passed: High logic preservation.")
        else:
            print(
                "WARNING: Significant divergence between "
                "PyTorch and ONNX models."
            )

        return self

    def validate(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(
            self.onnx_file_path, providers=providers
        )
        input_name = ort_session.get_inputs()[0].name

        self.datamodule.setup(stage="validate")
        val_loader = self.datamodule.val_dataloader()
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        correct, total = 0, 0

        for images, labels in tqdm(val_loader, desc="ONNX Validation"):
            images_np = images.cpu().numpy()
            labels_np = labels.cpu().numpy()

            logits = ort_session.run(None, {input_name: images_np})[0]
            predictions = np.argmax(logits, axis=1)
            correct += np.sum(predictions == labels_np)
            total += len(labels_np)

        accuracy = (correct / total) * 100
        return accuracy
