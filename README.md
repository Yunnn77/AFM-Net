<p align="center">
  <img src="docs/logo.png" alt="AFM-Net Logo" width="150"/>
</p>

<h1 align="center">AFM-Net: Advanced Fusion Model Network</h1>

<p align="center">
  <strong>Authors:</strong><br>
  Yuanhao Tang, Zhengpei Hu (Graduate Student Member, IEEE)<br>
  Junliang Xing (Senior Member, IEEE)<br>
  Chengkun Zhang, Jianqiang Huang (Member, IEEE)<br>
  <a href="https://github.com/tangyuanhao-qhu/AFM-Net">GitHub Repository</a>
</p>

---

![License](https://img.shields.io/badge/License-MIT-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

---

## Overview

AFM-Net is a dual-branch framework for **remote sensing scene classification** of high-resolution images.  
It combines the hierarchical visual priors of a CNN backbone with the global sequence modeling capability of Vision Mamba.  
A multi-scale fusion strategy enhances cross-branch feature integration, and a Mixture-of-Experts (MoE) classifier adaptively aggregates the most informative features.

**Key Features:**
- Dual-branch architecture (CNN + Vision Mamba)
- Multi-scale feature fusion
- Mixture-of-Experts (MoE) classifier
- State-of-the-art performance on AID, NWPU-RESISC45, and UC Merced datasets

---

## Model Architecture

<p align="center">
  <img src="docs/AFM-Net_architecture.png" alt="AFM-Net Architecture" width="700"/>
</p>

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tangyuanhao-qhu/AFM-Net.git
cd AFM-Net
2.	Create a Python virtual environment and install dependencies:
conda create -n afm-net python=3.8 -y
conda activate afm-net
pip install -r requirements.txt
3. Usage
Training
python train.py --dataset AID --batch_size 32 --epochs 500
4. Citation
@article{tang2025afmnet,
  title={AFM-Net: Advanced Fusion Model Network for Remote Sensing Scene Classification},
  author={Tang, Yuanhao and Hu, Zhengpei and Xing, Junliang and Zhang, Chengkun and Huang, Jianqiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}


