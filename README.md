![AFM-Net Logo](docs/logo.png)

# AFM-Net: Advanced Fusion Model Network

![License](https://img.shields.io/badge/License-MIT-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

**Authors:**  
Yuanhao Tang, Zhengpei Hu (Graduate Student Member, IEEE),  
Junliang Xing (Senior Member, IEEE),  
Chengkun Zhang, Jianqiang Huang (Member, IEEE)  

GitHub Repository: [https://github.com/tangyuanhao-qhu/AFM-Net](https://github.com/tangyuanhao-qhu/AFM-Net)

AFM-Net is a dual-branch framework for **remote sensing scene classification** of high-resolution images. It combines the hierarchical visual priors of a CNN backbone with the global sequence modeling capability of Vision Mamba. A multi-scale fusion strategy enhances cross-branch feature integration, and a Mixture-of-Experts (MoE) classifier adaptively aggregates the most informative features.

**Key Features:**
- Dual-branch architecture (CNN + Vision Mamba)
- Multi-scale feature fusion
- Mixture-of-Experts (MoE) classifier
- State-of-the-art performance on:
  - AID
  - NWPU-RESISC45
  - UC Merced

Model architecture:

![AFM-Net Architecture](docs/fig1.png)

## Installation

**Clone this repository:**
```bash
git clone https://github.com/tangyuanhao-qhu/AFM-Net.git
cd AFM-Net
