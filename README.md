# zkCNN - Python-centric Workflow

## Introduction

This project implements a GKR-based Zero-Knowledge proof system for Convolutional Neural Networks (CNNs), as described in [this paper](https://eprint.iacr.org/2021/673). It features common CNN models such as LeNet5 and provides the capability to generate and verify proofs for model inferences.

This refactored version provides a more elegant and user-friendly experience by offering a Python-centric workflow. Users can define and train models using PyTorch, and the system automatically handles the conversion to the C++ cryptographic backend for proof generation and verification.

**Note:** Currently, this version focuses on efficient proof generation and verification of CNN inferences and does not yet add complete zero-knowledge property or full training within the ZK context.

## Requirements

To run this project, you will need:

*   **Python 3.8+**
    *   `torch`
    *   `torchvision`
    *   `einops`
    *   `numpy`
*   **C++14** compiler (e.g., g++ or clang++)
*   **cmake >= 3.10**
*   **GMP library** (for C++ dependencies)
*   **Git** (for cloning submodules)

## Project Structure

The codebase is structured for clarity and maintainability:

*   `zkcnn/`: The main Python package.
    *   `zkcnn/__init__.py`: Package initialization.
    *   `zkcnn/core.py`: Contains the `ZKProver` class for managing the C++ backend (build, execution).
    *   `zkcnn/models.py`: Defines PyTorch CNN models (e.g., `LeNetCifar`).
    *   `zkcnn/converters.py`: Handles the export of PyTorch model weights and inputs to the C++-compatible format, leveraging `einops` for efficient tensor manipulation.
*   `src/`: C++ source code for the ZK proof system.
    *   `src/zkcnn_cli.cpp`: A new, unified C++ command-line interface that accepts model types and parameters dynamically. This replaces the old `main_demo_*.cpp` files.
    *   Other `.cpp` and `.h/.hpp` files implement the core cryptographic and neural network logic in C++.
*   `run_zkcnn.py`: A high-level Python script that orchestrates the entire workflow: PyTorch training, model export, and invocation of the C++ prover/verifier.
*   `script/`: Contains helper shell scripts, primarily `build.sh` for compiling the C++ backend.
*   `3rd/`: Third-party dependencies, including `hyrax-bls12-381` (a polynomial commitment scheme).

## Getting Started

### 1. Clone the repository and initialize submodules:

```bash
git clone --recurse-submodules git@github.com:TAMUCrypto/zkCNN.git
cd zkCNN
```
**Note**: If you cloned without `--recurse-submodules`, you can initialize them later:
```bash
git submodule update --init --recursive
```

### 2. Install Python dependencies:

```bash
pip install torch torchvision einops numpy
```

### 3. Run a demo:

The `run_zkcnn.py` script handles everything from PyTorch training (on synthetic data for demo purposes) to C++ proof generation and verification. It automatically builds the C++ backend if necessary.

**Run LeNet on CIFAR-10:**
```bash
python3 run_zkcnn.py CIFAR10
```

**Run LeNet on CIFAR-100:**
```bash
python3 run_zkcnn.py CIFAR100
```

The script will output the PyTorch prediction, trigger the C++ prover, and then display the zkCNN inference result, comparing it to the PyTorch prediction.

## Input Format (handled by `zkcnn/converters.py`)

The Python `zkcnn.converters.export_model` function now handles the preparation of input data and model weights into a format expected by the C++ backend (`src/zkcnn_cli.cpp`). This involves flattening image data, convolution kernels, and fully-connected layer weights/biases into a sequential list of doubles.

A dummy `config_file` is created as the C++ legacy argument parsing still expects it, but its content is not used in the current implementation.

## Polynomial Commitment

This project utilizes a [Hyrax polynomial commitment](https://eprint.iacr.org/2017/1132.pdf) based on the BLS12-381 elliptic curve, integrated as a submodule (`3rd/hyrax-bls12-381`).

## Reference

*   [zkCNN: Zero knowledge proofs for convolutional neural network predictions and accuracy](https://doi.org/10.1145/3460120.3485379). Liu, T., Xie, X., & Zhang, Y. (CCS 2021).
*   [Doubly-efficient zksnarks without trusted setup](https://doi.org/10.1109/SP.2018.00060). Wahby, R. S., Tzialla, I., Shelat, A., Thaler, J., & Walfish, M. (S&P 2018).
*   [Hyrax](https://github.com/hyraxZK/hyraxZK.git)
*   [mcl](https://github.com/herumi/mcl)