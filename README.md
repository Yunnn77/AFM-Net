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
| Model                      | Param (M) | UC Merced (P) | UC Merced (R) | UC Merced (F1) | AID (P) | AID (R) | AID (F1) | NWPU (P) | NWPU (R) | NWPU (F1) |
| :------------------------- | :-------: | :-----------: | :-----------: | :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: |
| **CNN-based Models**       |           |               |               |                |         |         |          |          |          |          |
| ResNet18                   | 11.2      | 90.40         | 90.32         | 90.22          | 92.28   | 92.24   | 92.22    | 93.82    | 93.75    | 93.75    |
| ResNet50                   | 23.6      | 91.25         | 90.95         | 90.85          | 92.34   | 92.26   | 92.22    | 94.44    | 94.40    | 94.40    |
| ResNet101                  | 42.6      | 93.93         | 93.81         | 93.74          | 91.87   | 91.80   | 91.77    | 94.80    | 94.75    | 94.75    |
| **Transformer-based Models**|           |               |               |                |         |         |          |          |          |          |
| DeiT-T                     | 5.5       | 85.07         | 84.44         | 84.42          | 80.74   | 80.72   | 80.63    | 83.57    | 83.57    | 83.45    |
| DeiT-S                     | 21.7      | 91.90         | 91.75         | 91.61          | 80.98   | 81.04   | 80.92    | 83.12    | 82.99    | 82.98    |
| DeiT-B                     | 85.8      | 92.53         | 92.38         | 92.34          | 82.20   | 82.14   | 82.05    | 80.11    | 80.08    | 79.98    |
| ViT-B                      | 88.3      | 90.49         | 90.32         | 90.18          | 82.79   | 82.72   | 82.54    | 80.25    | 80.26    | 80.16    |
| ViT-L                      | 303.0     | 92.80         | 92.54         | 92.42          | 82.08   | 81.96   | 81.84    | 79.50    | 79.56    | 79.46    |
| Swin-T                     | 27.5      | 89.28         | 88.89         | 88.88          | 87.41   | 87.40   | 87.35    | 89.79    | 89.75    | 89.72    |
| Swin-S                     | 48.9      | 90.01         | 89.84         | 89.72          | 87.03   | 86.98   | 87.03    | 89.42    | 89.28    | 89.28    |
| Swin-B                     | 86.8      | 91.93         | 91.75         | 91.66          | 86.51   | 86.44   | 86.37    | 89.55    | 89.40    | 89.41    |
| **Mamba-based Models**     |           |               |               |                |         |         |          |          |          |          |
| ViMamba-T                  | 30.0      | 93.14         | 92.85         | 92.81          | 91.59   | 90.94   | 91.10    | 93.97    | 93.96    | 93.94    |
| VisionMamba-T              | 7.1       | 83.83         | 83.81         | 83.06          | 79.16   | 78.94   | 78.68    | 89.24    | 89.02    | 88.97    |
| VisionMamba-S              | 25.6      | 89.62         | 89.68         | 89.32          | 87.77   | 87.66   | 87.54    | 95.23    | 95.22    | 95.21    |
| VisionMamba-B              | 96.9      | 88.94         | 89.05         | 88.82          | 90.98   | 90.80   | 90.72    | 95.10    | 95.07    | 95.06    |
| RSMamba-B                  | 6.4       | 94.14         | 93.97         | 93.88          | 92.02   | 91.53   | 91.66    | 94.87    | 94.87    | 94.84    |
| RSMamba-L                  | 16.2      | 95.03         | 94.76         | 94.74          | 92.31   | 91.75   | 91.90    | 95.03    | 95.05    | 95.02    |
| RSMamba-H                  | 33.1      | 95.47         | 95.23         | 95.25          | 92.97   | 92.51   | 92.63    | 95.22    | 95.19    | 95.18    |
| HC-Mamba-T                 | 6.6       | 94.12         | 94.59         | 94.76          | 91.97   | 91.47   | 91.42    | 94.88    | 94.96    | 94.87    |
| HC-Mamba-S                 | 15.7      | 95.10         | 95.00         | 95.08          | 92.33   | 91.88   | 91.95    | 95.10    | 95.12    | 95.08    |
| HC-Mamba-B                 | 32.8      | 95.55         | 95.31         | 95.34          | 93.02   | 92.68   | 92.86    | 95.32    | 95.26    | 95.25    |
| **Ours (AFM-Net)**         | **45.5**  | **96.92**     | **96.83**     | **96.81**      | **93.76**| **93.72**| **93.71** | **95.54** | **95.52** | **95.52** |

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










