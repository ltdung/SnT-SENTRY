# SnT-SENTRY
The SENTRY project aims to optimize Earth Observation (EO) systems by integrating cutting-edge satellite communication networks with novel communication strategies. It is funded by the Luxembourg National Research Fund (FNR) and led by the SIGCOM group at the Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxembourg.

For more FNR research projects, led by SIGCOM, please visit: https://www.uni.lu/snt-en/research-groups/sigcom/research/

# Task 1 - EO Image Classification
It aims to explore the most effective, lightweight, pre-trained ViT model that can be employed in onboard satellites.

## Requirements to Reproduce the Code:
To reproduce the experiment, it is highly recommended that you install the required packages from the exported environment file (.yml).

### To create a replicate env, please follow:

conda config --set channel_priority flexible

conda env create -f environment.yml

If it does not work: conda env create -f environment.yml --no-deps

### There are two different environments (one for MobileViT, and the other for EfficientViT)
MobileViT: environment_torch.yml

EfficientViT: environment_efficientViT.yml

## Data Preparation:
### Data Augmentation: 

data augmentation.ipynb

### Inference Noisy Data: 
Gaussian noise: noisy_data_gaussian.ipynb

Motion blur noise: inference_efficientViTM2_motionblur.ipynb

## Training Models
### Model Computational Complexity
computational complexity_MobileViTV2.ipynb

computational complexity_EfficientViT-M2.ipynb
### Statistical Performance
statistical_MobileViTV2.ipynb

statistical_EfficientViT-M2.ipynb


## Pretrained Weights and Inference Models:
### Pretrained weight
trained weight_MobileViT2.pth

trained weight_EfficientViT_M2.pth

### Inference Models
inference_MobileViT2_gaussian.ipynb

inference_MobileViT2_motionblur.ipynb

inference_efficientViTM2_gaussian.ipynb

inference_efficientViTM2_motionblur.ipynb

### Inference Power Consumption
inference_MobileViT_pc.ipynb

inference_efficientViT_M2_pc.ipynb






