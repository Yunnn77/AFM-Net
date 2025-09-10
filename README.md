AFM-Net: Advanced Fusion Model Network
![alt text](https://img.shields.io/badge/License-MIT-blue)
![alt text](https://img.shields.io/badge/Python-3.8%2B-green)
![alt text](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

Authors:
Yuanhao Tang, Zhengpei Hu (Graduate Student Member, IEEE)
Junliang Xing (Senior Member, IEEE)
Chengkun Zhang, Jianqiang Huang (Member, IEEE)

GitHub Repository: https://github.com/tangyuanhao-qhu/AFM-Net

Overview
AFM-Net is a dual-branch framework for remote sensing scene classification of high-resolution images. It combines the hierarchical visual priors of a CNN backbone with the global sequence modeling capability of Vision Mamba. A multi-scale fusion strategy enhances cross-branch feature integration, and a Mixture-of-Experts (MoE) classifier adaptively aggregates the most informative features.

Key Features:

Dual-branch architecture (CNN + Vision Mamba)

Multi-scale feature fusion

Mixture-of-Experts (MoE) classifier

State-of-the-art performance on:

AID

NWPU-RESISC45

UC Merced

Model Architecture
![alt text](docs/AFM-Net_architecture.png)

Installation
Clone this repository:

code
Bash
git clone https://github.com/tangyuanhao-qhu/AFM-Net.git
cd AFM-Net
Create and activate a Conda environment:

code
Bash
conda create -n afm-net python=3.8 -y
conda activate afm-net
Install dependencies:

code
Bash
pip install -r requirements.txt
Training
Run the following command to start training the model on the AID dataset with a batch size of 32 for 100 epochs.

code
Bash
python train.py --dataset AID --batch_size 32 --epochs 100
Citation
If you use this code for your research, please cite:

code
Bibtex
@article{tang2025afmnet,
  title={AFM-Net: Advanced Fusion Model Network for Remote Sensing Scene Classification},
  author={Tang, Yuanhao and Hu, Zhengpei and Xing, Junliang and Zhang, Chengkun and Huang, Jianqiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
