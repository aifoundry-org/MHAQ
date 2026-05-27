import argparse
import os
import resource
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.onnx import register_custom_op_symbolic
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from src.config.config_loader import load_and_validate_config
from src.data.compose.composer import DatasetComposer
from src.models.compose.composer import ModelComposer
from src.quantization.quantizer import Quantizer
from src.training.trainer import Trainer, Validator
from src.loggers.default_logger import logger

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="Run GDNSQ quantization.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the configuration file (YAML).",
        # default="config/gdnsq_config_resnet20_cifar100_ste_w1a1.yaml"
        # default="config/gdnsq_config_resnet20_old.yaml"
        default="config/gdnsq_config_resnet20_new.yaml"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        help="Path to resume/load checkpoint",
        default=None
    )
    return parser.parse_args()


class ONNXPipeline:
    def __init__(self, qmodel, datamodule, onnx_file_path):
        self.qmodel = qmodel
        self.datamodule = datamodule
        self.onnx_file_path = onnx_file_path

    def export(self):
        self.qmodel.eval()
        self.qmodel.cpu()

        dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)

        def custom_exp2_symbolic(g, x):
            two = g.op(
                "Constant", value_t=torch.tensor(2.0, dtype=torch.float32)
            )
            return g.op("Pow", two, x)

        register_custom_op_symbolic('aten::exp2', custom_exp2_symbolic, 15)

        torch.onnx.export(
            self.qmodel,
            dummy_input,
            self.onnx_file_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamo=False,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        return self

    def verify(self):
        onnx_model = onnx.load(self.onnx_file_path)
        onnx.checker.check_model(onnx_model)

        self.qmodel.eval()
        self.qmodel.cpu()

        torch.manual_seed(42)
        dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)

        with torch.no_grad():
            torch_output = self.qmodel(dummy_input).numpy()

        ort_session = ort.InferenceSession(self.onnx_file_path)
        input_name = ort_session.get_inputs()[0].name
        ort_output = ort_session.run(
            None, {input_name: dummy_input.numpy()}
        )[0]

        t_flat, o_flat = torch_output.flatten(), ort_output.flatten()
        cos_sim = np.dot(t_flat, o_flat) / (
            np.linalg.norm(t_flat) * np.linalg.norm(o_flat)
        )

        torch_pred = np.argmax(torch_output, axis=1)[0]
        ort_pred = np.argmax(ort_output, axis=1)[0]

        print(f"Cosine Similarity:  {cos_sim:.4f}")
        print(f"PyTorch Prediction: Class {torch_pred}")
        print(f"ONNX Prediction:    Class {ort_pred}")

        if torch_pred == ort_pred and cos_sim > 0.99:
            print("Verification Passed: Math and logic are intact.")
        else:
            if torch_pred != ort_pred:
                print("WARNING: Predictions differ.")
            if cos_sim <= 0.99:
                print("WARNING: Cosine similarity is low.")

        return self

    def validate(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(
            self.onnx_file_path, providers=providers
        )
        input_name = ort_session.get_inputs()[0].name

        self.datamodule.setup(stage="fit")
        val_loader = self.datamodule.val_dataloader()
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        correct, total = 0, 0

        for images, labels in tqdm(val_loader, desc="ONNX Validation"):
            logits = ort_session.run(None, {input_name: images.numpy()})[0]
            predictions = np.argmax(logits, axis=1)
            correct += np.sum(predictions == labels.numpy())
            total += len(labels.numpy())

        accuracy = (correct / total) * 100
        return accuracy


def main():
    args = parse_args()

    config = load_and_validate_config(args.config)
    dataset_composer = DatasetComposer(config=config)
    model_composer = ModelComposer(config=config)
    quantizer = Quantizer(config=config)()
    validator = Validator(config=config)
    trainer = Trainer(config=config)

    data = dataset_composer.compose()
    model = model_composer.compose()

    logger.info(f"Validate Model before quantization:\n{model}")
    validator.validate(model, datamodule=data)

    qmodel = quantizer.quantize(model, in_place=True)

    logger.info("Validate model after layers replacement")
    validator.validate(qmodel, datamodule=data)

    logger.info("Calibrating model initial weights and scales")
    validator.calibrate(qmodel, datamodule=data)

    qmodel.train()
    trainer.fit(qmodel, datamodule=data, ckpt_path=args.ckpt)

    validator.validate(qmodel, datamodule=data)

    pipeline = ONNXPipeline(
        qmodel=qmodel,
        datamodule=data,
        onnx_file_path="resnet20_cifar10_w1a1.onnx"
    )

    onnx_accuracy = pipeline.export().verify().validate()
    print(f"\nONNX Accuracy: {round(onnx_accuracy, 2):.2f}%")


if __name__ == "__main__":
    main()
