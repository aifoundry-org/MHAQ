# MHAQ: Moderately Hackable Quantization framework

## Introduction

This repository provides a customizable and automated environment for developing and testing different quantization methods.

## Features

- **Customizable Quantization:** The framework allows the definition and integration of custom quantization methods. It is designed to work seamlessly with pre-configured data pipelines.

- **Ease of Use:** Built on PyTorch Lightning, providing a streamlined API that simplifies model management and training, while leveraging Lightning’s features for distributed training and logging.

- **Broad Model Support:** Designed to support models across multiple domains, including Computer Vision, Natural Language Processing, and Audio, with flexibility for different architectures.

- **Modular and Hackable Design:** The framework's modular architecture makes it easy to modify and extend, enabling straightforward customization for specific needs.

## Installation

Instructions to setup framework:

1. Clone this repo with

     ```bash
    git clone https://github.com/aifoundry-org/Quant.git
    ```

2. Install project dependencies *(it is better to use wit virtual environment)*

    ```bash
    pip3 install -r requirements.txt
    ```

## Getting started with QAT

### 1. Define pipeline with config

*Config schema placed in `src/config/config_schema.py`*

```python
from src.config.config_loader import load_and_validate_config

config = load_and_validate_config("config/{PATH_TO_YOUR_CONFIG}")

```

### 2. Initialize model to quantize

```python
from src.models.compose.composer import ModelComposer

composer = ModelComposer(config=config)

model = composer.compose()
```

### 3. Initialize trainer

```python
from src.training.trainer import Trainer

trainer = Trainer(config=config)
```

### 4. Initialize quantizer

```python
from src.quantization.quantizer import Quantizer

quantizer = Quantizer(config=config)()
```

### 5. Define your data

```python
from src.data import YOURDATAMODULE

data = YOURDATAMODULE
```

### 6. Quantize and train

```python
qmodel = quantizer.quantize(model, in_place=True)

trainer.fit(qmodel, datamodule=data)
```

## Getting started with PTQ

Not yet implemented.

## Customization and Hackability

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md)

## Performance Metrics

### CIFAR-10
| Model     | Dataset  | Method | QW | QA | Best Top-1 |
|-----------|----------|--------|----|----|------------|
| Resnet-20 | CIFAR-10 | FP (finetuned)     | -  | -  | 92.82%      |
| Resnet-20 | CIFAR-10 | GDNSQ   | 1  | 1  | 85.78%      |
| Resnet-20 | CIFAR-10 | GDNSQ   | 1  | 32 | 92.14%      |   
| Resnet-20 | CIFAR-10 | GDNSQ   | 2  | 2  | 91.4%      |
| Resnet-20 | CIFAR-10 | GDNSQ   | 3  | 3  | 92.5%      |
| Resnet-20 | CIFAR-10 | GDNSQ   | 4  | 4  | 92.78%      |

### CIFAR-100
| Model     | Dataset   | Method | QW | QA | Best Top-1  |
|-----------|-----------|--------|----|----|-------------|
| Resnet-20 | CIFAR-100 | FP     | -  | -  | 70.26%      |
| Resnet-20 | CIFAR-100 | GDNSQ   | 1  | 1  | 56.94%      |
| Resnet-20 | CIFAR-100 | GDNSQ   | 2  | 2  | 66.41%      |
| Resnet-20 | CIFAR-100 | GDNSQ   | 3  | 3  | 69.36%      |
| Resnet-20 | CIFAR-100 | GDNSQ   | 4  | 4  | 70.24%      |

### IMAGENET-1K
| Model     | Dataset     | Method | QW | QA | Best Top-1  |
|-----------|-------------|--------|----|----|-------------|
| Resnet-18 | IMAGENET-1k | FP(finetuned)     | -  | -  | 71.91%      |
| Resnet-18 | IMAGENET-1k | FP(finetuned)     | 1  | 32  | 66.14%      |
