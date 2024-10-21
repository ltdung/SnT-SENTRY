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

## 

