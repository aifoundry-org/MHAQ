# Contributing to MHAQ

Thank you for your interest in contributing to MHAQ! We welcome contributions in the form of bug reports, feature requests, code, and documentation. Please read through the following guidelines to get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Style Guide](#style-guide)
- [Project Structure](#project-structure)
- [Add new quantization method](#add-quantization)


## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and considerate in your communications and actions.

## How to Contribute

1. **Fork the repository** and create a new branch for your changes.
2. **Clone the repository** to your local machine:

    ```bash
    git clone https://github.com/aifoundry-org/MHAQ.git
    ```

3. **Create a new branch** for your contribution:

    ```bash
    git checkout -b feature/your-feature-name
    ```

4. Make your changes and commit them with a descriptive message:

    ```bash
    git commit -m "Add feature: description of your feature"
    ```

5. **Push your changes** to your forked repository:

    ```bash
    git push origin feature/your-feature-name
    ```

6. Open a **Pull Request** to the main repository.

## Setting Up the Development Environment

1. Clone the repository:

    ```bash
    git clone https://github.com/aifoundry-org/MHAQ.git
    cd MHAQ
    ```

2. Create a virtual environment:

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Reporting Bugs

If you find a bug, please report it by opening an issue on GitHub. Include as much detail as possible:

- **Steps to reproduce** the bug.
- **Expected behavior** and what actually happened.
- Your **environment** (Python version, OS, etc.).

## Suggesting Features

We welcome feature requests! Please open an issue and provide:

- A **clear description** of the feature.
- **Use cases** for the feature.
- If possible, a **proposed implementation**.

## Pull Request Guidelines

1. Ensure that your code adheres to the [Style Guide](#style-guide).
2. Include tests that cover the changes you made.
3. Make sure all tests pass before submitting your pull request.
4. Link your pull request to the related issue.

## Style Guide

Follow the PEP 8 coding style. Use flake8 to check for style issues:

```bash
flake8 your_module/
```

## Project Structure

Execute ```git ls-files | tree --fromfile -d```

```bash
Quant 
├── config  # for training/quant config files
├── data    # for datasets placement
├── scripts # for scripts utilizing the main functionality
├── src
│   ├── aux       # for additional funcs / small utils
│   ├── callbacks # for lightning callbacks
│   ├── config    # for configs handlers
│   ├── data      # for data processing
│   │   └── vision_cls
│   ├── models    # for models definition
│   │   └── compose
│   │       └── vision
│   ├── quantization # for quantization methods definition
│   │   ├── abc
│   │   ├── dummy
│   │   └── rniq
│   └── training    # for specific training regimes
└── tests           # for tests
```

## Add Quantization

1. In order to add your quantization method, you should create new folder in `src/quantization` folder named after your method.
2. It is convinient to inherit your quantization class from `BaseQuant` at `src/quantization/abc/abc_quant.py`
and redefine the abstract methods. However, it's not mandatory and `BaseQuant`
class may not satisfy your needs.
3. You can define schema for your quantization config in `src/config/config_schema.py`
4. After that you should be able to wrap model with your quantization approach
and perform training/tuning.

*You can look into implemented Dummy and RNIQ methods to understand it further.*