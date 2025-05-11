# 🚀 Contrastive Learning on CIFAR-10 with PyTorch
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![CI](https://img.shields.io/badge/CI-passing-brightgreen)

Advanced self-supervised learning implementation featuring ResNet architectures, NT-Xent loss, and comprehensive visualization tools.

## 📌 Table of Contents
- [Features](#-features)
- [Installation](#-installation)  
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Visualization](#-visualization)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Features
- 🛠 Custom augmentation pipeline (8 transformations)
- 🧠 Supports ResNet18/34/50 backbones
- 📊 Real-time metric tracking (loss, similarity)
- 🎨 t-SNE/PCA visualization tools
- 🔍 Image similarity analysis
- 🕵 Odd-one-out detection
- ⚡ Multi-GPU training support

## 💻 Installation
# Clone repository
git clone https://github.com/yourusername/contrastive-learning.git
cd contrastive-learning

# Install dependencies (Python 3.8+ required)
pip install torch torchvision numpy matplotlib scikit-learn tqdm pillow pytorch-lightning

## 🚦 Quick Start
# Train default model (ResNet18)
python train.py --backbone resnet18 --epochs 20 --batch_size 256

# Generate embeddings visualization
python visualize.py --model_path runs/checkpoint.pth --output_dir results/

## 🏋 Training
### Configuration Options
| Argument | Default | Description |
|----------|---------|-------------|
| --backbone | resnet18 | Architecture (resnet18/34/50) |
| --epochs | 20 | Training epochs |
| --batch_size | 256 | Batch size |
| --temperature | 0.1 | NT-Xent loss temperature |
| --lr | 1e-3 | Learning rate |

### Example Training Command
python train.py \
    --backbone resnet50 \
    --epochs 50 \
    --batch_size 512 \
    --lr 5e-4 \
    --temperature 0.07 \
    --gpus 2

## 📈 Evaluation
### Expected Metrics
| Epoch | Loss | PosSim | NegSim | Notes |
|-------|------|--------|--------|-------|
| 1 | 4.123 | 0.65 | 0.15 | Warmup phase |
| 10 | 2.456 | 0.78 | 0.08 | Learning progressing |
| 20 | 1.892 | 0.85 | 0.03 | Convergence |

## 🎨 Visualization
### Available Tools
#### Embedding Projection
python visualize.py --model_path model.pth --method tsne --perplexity 30

#### Image Similarity
python similarity.py --image1 cat.jpg --image2 dog.jpg

#### Odd-One-Out Detection
python odd_one_out.py --folder images/ --top_k 5

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See LICENSE for more information.

## 📧 Contact
Your Name - your.email@example.com

Project Link: https://github.com/yourusername/contrastive-learning
