# Contrastive Learning on CIFAR-10 with PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project implements contrastive self-supervised learning methods for visual representation learning on the CIFAR-10 dataset using PyTorch.

## Table of Contents
- [Key Features](#key-features)
- [Implemented Methods](#implemented-methods)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

## Key Features

✔ Implementation of popular contrastive learning frameworks (SimCLR, MoCo)  
✔ Data augmentation pipeline for contrastive learning  
✔ Customizable projection heads and backbone architectures  
✔ Visualization of learned embeddings using t-SNE  
✔ Linear evaluation protocol to assess representation quality  

## Implemented Methods

- *SimCLR* (A Simple Framework for Contrastive Learning)
- *MoCo v2* (Momentum Contrast)
- *BYOL* (Bootstrap Your Own Latent) - Coming soon

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- CUDA Toolkit (recommended for GPU acceleration)
- Other dependencies: numpy, matplotlib, scikit-learn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/contrastive-learning-cifar10.git
cd contrastive-learning-cifar10
