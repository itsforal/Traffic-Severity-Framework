# 🚦 Traffic Crash Severity Prediction Framework
### Hierarchical Deep Learning (DeepInsight) & Classical ML Ensemble
**Reproduction and Extension of Rahim & Hassan (2021)**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Research](https://img.shields.io/badge/Reproduction-Rahim_%26_Hassan_2021-green.svg)](https://doi.org/10.1016/j.aap.2021.106090)

---

## Project Overview
This repository **explores** a machine learning framework for classifying traffic accident severity using the **US-Accidents dataset (7.7M records)**. The project aims to investigate the feasibility of transforming tabular data into image formats for severity prediction.

### Key Highlights:
- **Phase 1: Classical ML Ensemble** – Implementation of Gradient Boosting (GBM), Random Forest, and MLP baselines.
- **Phase 2: DeepInsight Methodology** – Applying a transformation technique that maps tabular data into 2D images via **t-SNE** to utilize the spatial feature extraction capabilities of **CNNs (EfficientNet-B0)**.
- **Hierarchical Inference** – A two-stage model structure designed to address extreme class imbalance.
- **Recall Focus** – Experimenting with a custom **Soft F-Beta Loss** to prioritize the detection of fatal crashes ($\beta=2.0$).

---

## Dataset Information
The project utilizes the **US Accidents (2016-2023)** database. 

⚠️ **Note:** Due to its large size (~3.5 GB), the raw CSV is not included in this repository.

### Data Setup Instructions:
1. **Download:** Get the `US_Accidents_March23.csv` from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
2. **Directory:** Create a folder named `data/` in the project root.
3. **Placement:** Move the downloaded file into the `data/` folder.

---

## Methodology

### 1. Numeric-to-Image Transformation (DeepInsight)
We map 1D feature vectors into a $120 \times 120$ pixel grid. Correlated features (e.g., Temperature, Humidity) are clustered together using **t-SNE**, creating "feature regions" that allow CNNs to process tabular relationships spatially.

### 2. Hierarchical Model Architecture
To handle multi-class imbalance:
- **Stage 1 (Fatal Detection):** Binary classifier for *Fatal* (Severity 4) vs. *Non-Fatal*.
- **Stage 2 (Injury Severity):** Distinguishes *Injury* (Severity 3) vs. *PDO* (Severity 1-2).

### 3. Custom Soft F-Beta Loss
To address the rarity of fatal accidents, we implemented a differentiable loss function:
$$Loss = 1 - \frac{(1 + \beta^2) \cdot TP}{(1 + \beta^2) \cdot TP + \beta^2 \cdot FN + FP}$$
*Setting $\beta > 1$ penalizes False Negatives (missing a fatal crash) more heavily than False Positives.*

---

## Experimental Results
The experiments highlight the trade-off between Precision and Recall when tuning the $\beta$ parameter for safety-critical tasks.

| Model Phase | Beta ($\beta$) | Recall | Precision | F1-Score | Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: Fatal** | 2.0 | **1.00** | 0.33 | 0.50 | **Safety-Critical** |
| **Stage 1: Fatal** | 1.0 | 0.77 | 0.43 | 0.55 | Balanced |
| **Stage 2: Injury** | 2.0 | **1.00** | 0.53 | 0.69 | Recall-Oriented |

### Interpretations
At $\beta=2.0$, the model successfully identifies **100%** of fatal cases (Recall). While this introduces more False Alarms (lower Precision), this behavior is often preferred in safety contexts where missing a fatal crash is the worst-case scenario.

---

## Repository Structure
```text
├── data/               # (Gitignored) Raw US-Accidents CSV
├── outputs/            # Generated charts and confusion matrices
├── src/                # Logic Core
│   ├── config.py       # Centralized hyperparameters
│   ├── preprocess.py   # ETL & memory optimization
│   ├── dl_engine.py    # DeepInsight & EfficientNet architecture
│   └── utils.py        # Soft F-Beta Loss & visualization helpers
├── main.py             # Application Entry Point
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
