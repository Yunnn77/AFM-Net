# AFM-Net: Advanced Fusion Model Network for Remote Sensing Scene Classification
<p align="center">
  <img src="docs/logo.png" alt="AFM-Net Logo" width="200"/>
</p>

<h1 align="center">AFM-Net: Advanced Fusion Model Network</h1>

---

![License](https://img.shields.io/badge/License-MIT-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

---

##  Introduction 🔍 

Remote sensing scene classification of high-resolution images remains a challenging task due to the complex spatial structures and multi-scale characteristics of objects.
We propose AFM-Net (Advanced Fusion Model Network), a dual-branch framework that:
* ✅ Combines CNN backbone (local texture & hierarchical priors) with Vision Mamba (global sequence modeling).
* ✅ Employs a multi-scale fusion strategy for cross-branch feature integration.
* ✅ Uses a Mixture-of-Experts (MoE) classifier for adaptive feature aggregation.

Extensive experiments show AFM-Net achieves state-of-the-art performance on AID, NWPU-RESISC45, and UC Merced datasets.

---

## Model Architecture 📐 

<p align="center">
  <img src="docs/fig1.png" alt="AFM-Net Architecture" width="700"/>
</p>

## Installation ⚙️
### Clone this repository:
```bash
git clone https://github.com/tangyuanhao-qhu/AFM-Net.git
cd AFM-Net
```
### Create a Python virtual environment and install dependencies:
```bash
conda create -n afm-net python=3.8 -y
conda activate afm-net
pip install -r requirements.txt
```
## Usage 🚀
Training 🔹 
 ```bash
python train.py --dataset AID --batch_size 32 --epochs 100
 ```
## Results 📊 
Dataset
OA (%)
Improvement
AID
93.72
+0.74
NWPU-RESISC45
95.54
+0.22
UC Merced
96.92
+1.37
## Citation 📖 
If you use AFM-Net in your research, please cite our paper:
@article{tang2025afmnet,
  title={AFM-Net: Advanced Fusion Model Network for Remote Sensing Scene Classification},
  author={Tang, Yuanhao and Hu, Zhengpei and Xing, Junliang and Zhang, Chengkun and Huang, Jianqiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
## Authors 👨‍💻 
* Yuanhao Tang, Zhengpei Hu (Graduate Student Member, IEEE)
* Junliang Xing (Senior Member, IEEE)
* Chengkun Zhang, Jianqiang Huang (Member, IEEE)










