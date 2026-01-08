# 🚦 Traffic Crash Severity Prediction Framework
### Hierarchical Deep Learning (DeepInsight) & Classical ML Ensemble

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Reproduction-Rahim_%26_Hassan_2021-green.svg)](https://doi.org/10.1016/j.aap.2021.106090)

## 📌 Project Overview
This repository implements a high-performance framework for classifying traffic accident severity using the **US-Accidents dataset (7.7M records)**. It features a novel **Numeric-to-Image transformation** (DeepInsight) alongside traditional Gradient Boosting models.

## 🛠️ Key Methodologies
### 1. Classical Machine Learning (Stage A)
- **Models:** GBM, Random Forest, SVM, MLP, and XGBoost.
- **Optimization:** Memory-efficient preprocessing (40% RAM reduction) and GridSearch CV.

### 2. DeepInsight Deep Learning (Stage B)
Reproduction of *Rahim & Hassan (2021)*:
- **Feature Mapping:** 1D tabular vectors transformed to 2D image matrices via **t-SNE**.
- **Architecture:** Pre-trained **EfficientNet-B0** adapted for grayscale feature-images.
- **Proposed Tweak:** Hierarchical classification (Fatal vs. Non-Fatal) using a custom **Soft F-Beta Loss**.

## 📊 Results Summary
The model was evaluated across multiple $\beta$ values to prioritize **Recall** for fatal accidents.

| Model Phase | Beta ($\beta$) | Recall | Precision | F1-Score | Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: Fatal** | 2.0 | **1.00** | 0.33 | 0.50 | **Safety-Critical** |
| **Stage 1: Fatal** | 1.0 | 0.77 | 0.43 | 0.55 | Balanced |
| **Stage 2: Injury** | 2.0 | **1.00** | 0.53 | 0.69 | Robust Injury Detection |

### 📈 Interpretation
Increasing $\beta$ to 2.0 allowed the framework to achieve **100% Recall** for fatal crashes, ensuring no high-severity incidents are missed—a crucial requirement for Intelligent Transportation Systems (ITS).

## 📂 Repository Structure
- `src/preprocess.py`: Advanced data pipeline & memory optimization.
- `src/dl_engine.py`: DeepInsight (t-SNE) mapping & EfficientNet logic.
- `src/utils.py`: Implementation of the custom differentiable **Soft F-Beta Loss**.

## 🚀 Getting Started
```bash
git clone https://github.com/yourusername/Traffic-Severity-Framework.git
pip install -r requirements.txt
python main.py