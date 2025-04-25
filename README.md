# SnT-SENTRY
The SENTRY project aims to optimize Earth Observation (EO) systems by integrating cutting-edge satellite communication networks with novel communication strategies and machine learning approaches. It is funded by the Luxembourg National Research Fund (FNR) and led by the SIGCOM group at the Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxembourg.

For more FNR research projects, led by SIGCOM, please visit: https://www.uni.lu/snt-en/research-groups/sigcom/research/

# Task 1 - ViTs for Onboard Satellite-Based EO Image Classification (EO-IC)
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
### Dataset:
**EuroSAT**: Land Use and Land Cover Classification with Sentinel-2
https://github.com/phelber/EuroSAT

![eurosat_overview_small](https://github.com/user-attachments/assets/c3fefb53-3379-46e9-82db-15282795a9f5)

**PatternNet**: A benchmark dataset for performance evaluation of remote sensing image retrieval
https://sites.google.com/view/zhouwx/dataset

![image](https://github.com/user-attachments/assets/9e8b4d7e-73c7-43bb-9ef2-c1586151717c)



### Data Augmentation: 

data augmentation.ipynb

![augment](https://github.com/user-attachments/assets/2bdc9f92-731b-41c5-ad45-95373b5dae98)


### Inference on Unseen Data with Noisy (Gaussian, Motion Blur): 

![noise_level](https://github.com/user-attachments/assets/ce54dce4-2de4-44e3-ab16-df1c187dcba5)


## Training Models
### Model Computational Complexity
![complexity](https://github.com/user-attachments/assets/4aab05eb-aec7-44e0-a95f-d3ef3bb75253)

### Inference Power Consumption (W) Comparision
![power_consumption](https://github.com/user-attachments/assets/75e06312-c564-4772-8b23-ddfb00f731fa)

### Cite
```
@article{le2024board,
  title={On-board satellite image classification for earth observation: A comparative study of ViT models},
  author={Le, Thanh-Dung and Ha, Vu Nguyen and Nguyen, Ti Ti and Eappen, Geoffrey and Thiruvasagam, Prabhu and Garces-Socarras, Luis M and Chou, Hong-fu and Gonzalez-Rios, Jorge L and Merlano-Duncan, Juan Carlos and Chatzinotas, Symeon},
  year={2024}
  eprint={2409.03901},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2409.03901},
}
```

# Task 2 - Semantic Knowledge Distillation for EO-IC
The objective is to explore the most effective lightweight ResNet model for onboard satellite deployment by applying dynamic weighting in a semantic dual-teacher knowledge distillation (DualKD)  framework.

![KD_workflow](https://github.com/user-attachments/assets/ef5a9d3d-4b19-4ec3-8c84-80e04be73395)

We propose a dynamic weighting strategy in a semantic DualKD framework to enhance the lightweight ResNet8 model, achieving over 90% accuracy, precision, and recall. ResNet8, in particular, demonstrates substantial efficiency gains, with 97.5% fewer parameters, 96.7% fewer FLOPs, 86.2% lower power consumption, and 63.5% faster inference time compared to MobileViT. This significant reduction in complexity and resource requirements makes ResNet8 an optimal choice for EO-IC tasks, balancing high performance with practical deployment demands.

![acc_performance](https://github.com/user-attachments/assets/2cca0101-f02f-469b-ae5f-7ecd68f4275c)


### Cite
```
@article{le2024semantic,
  title={Semantic Knowledge Distillation for Onboard Satellite Earth Observation Image Classification},
  author={Le, Thanh-Dung and Ha, Vu Nguyen and Nguyen, Ti Ti and Eappen, Geoffrey and Thiruvasagam, Prabhu and Garces-Socarras, Luis M and Chou, Hong-fu and Gonzalez-Rios, Jorge L and Merlano-Duncan, Juan Carlos and Chatzinotas, Symeon},
  year={2024}
  eprint={2411.00209},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.00209},
}
```

# Task 3 - Enhanced Channel-Wise Adaptive Gated Linear Units SE for EO-IC

This study introduces ResNet-GLUSE, a lightweight ResNet variant enhanced with Gated Linear Unit-enhanced Squeeze-and-Excitation (GLUSE), an adaptive channel-wise attention mechanism. By integrating dynamic gating into the traditional SE framework, GLUSE improves feature recalibration while maintaining computational efficiency.

![image](https://github.com/user-attachments/assets/daa3006d-c6b4-4ad7-b2e0-7d4d34692ff5)

Our experimental results on the EuroSAT and PatternNet datasets demonstrate the model’s capability to achieve above 94% and 98% accuracy, respectively. While other models like MobileViT can reach 99% accuracy, our proposed ResNet-GLUSE shows remarkable efficiency, requiring 33× fewer parameters, 27x fewer FLOPs, 33x smaller model size (MB), approximately 6x lower power consumption (W) on GPU, and about 3x faster inference time (s). This drastic reduction in resource utilization positions ResNet-GLUSE as an ideal candidate for onboard satellite deployments, where real-time analysis and limited computational resources are critical. Moreover, its simple design makes it easily adaptable for neuromorphic computing, showing an ultra-low power inference of 852.30 mW on the Akida Brainchip platform.

![image](https://github.com/user-attachments/assets/41a5cfdb-008a-4a84-b2b4-3f519d6ccfc9)

![image](https://github.com/user-attachments/assets/c36f7d0c-2b4b-46fd-bc30-ea861617fca6)

![image](https://github.com/user-attachments/assets/9b9f9106-e626-4a84-86d8-287177d54b9f)
![image](https://github.com/user-attachments/assets/88f78fc6-510e-457b-bf75-881b42f4c4eb)

### Cite
```
@article{le2025gluse,
  title={GLUSE: Enhanced Channel-Wise Adaptive Gated Linear Units SE for Onboard Satellite Earth Observation Image Classification},
  author={Le, Thanh-Dung and Ha, Vu Nguyen and Nguyen, Ti Ti and Eappen, Geoffrey and Thiruvasagam, Prabhu and Chou, Hong-fu and Tran, Duc-Dung and Nguyen-Kha, Hung and Garces-Socarras, Luis M and Chou, Hong-fu and Gonzalez-Rios, Jorge L and Merlano-Duncan, Juan Carlos and Chatzinotas, Symeon},
  year={2025}
  eprint={2504.12484},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.12484},
}
```
# Task 4 - Task-Oriented Integration of Sensing, Computation, and Communication

EO satellites generate massive volumes of data that must be transmitted to ground stations for processing and analysis. This process faces several key challenges:

## Bandwidth Limitations: 
Satellite communication links have restricted bandwidth, making transmitting large volumes of raw EO data challenging.

## Latency Requirements: 
Many EO applications require timely data delivery, necessitating efficient transmission systems with minimal delay.

## Inefficient Traditional Communication: 
Conventional bit-oriented communication methods often transmit redundant and irrelevant information, wasting valuable bandwidth.

## Trade-offs Between Quality and Transmission Efficiency: 
Compression techniques reduce data volume but can degrade image quality, potentially affecting downstream task performance.

The study identify a significant gap in understanding how communication conditions affect the performance of EO applications. Current approaches lack models linking EO objectives to transmitted data, which is essential for optimizing communication systems for specific tasks, by following steps:

## Transmission System Architecture: 
The complete system architecture is shown below, demonstrating the flow from image compression to satellite transmission through DVB-S2(X) and eventual reception at the ground station.

![image](https://github.com/user-attachments/assets/6a5c5c72-ccb4-475f-bef1-665369eeec36)

## Transmission Loss Simulation: 
A parameter 's' representing the ratio of actual SNR to Shannon-based SNR is introduced to model transmission loss due to wireless constraints.

![image](https://github.com/user-attachments/assets/986b7346-8a01-4399-948c-13ce10088545)

The results obtained for semantic loss modeling of EfficientViT performance based on image quality and SNR ratio

![image](https://github.com/user-attachments/assets/7d806592-3b1f-4dfe-a604-30ef88e10fe1)

Mean Absolute Percentage Error (MAPE) versus number of terms in the fitting model for different ML architectures, showing significant error reduction with more terms.

![image](https://github.com/user-attachments/assets/d013fba9-c499-40cf-9b4f-c38ede121490)


### Cite
```
@article{nguyen2025semantic,
  title={A Semantic-Loss Function Modeling Framework With Task-Oriented Machine Learning Perspectives},
  author={Nguyen, Ti Ti and  Le, Thanh-Dung and Ha, Vu Nguyen and Eappen, Geoffrey and Thiruvasagam, Prabhu and Chou, Hong-fu and Tran, Duc-Dung and Nguyen-Kha, Hung and Garces-Socarras, Luis M and Chou, Hong-fu and Gonzalez-Rios, Jorge L and Merlano-Duncan, Juan Carlos and Chatzinotas, Symeon},
  year={2025}
  eprint={2503.09903},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.09903},
}
```


