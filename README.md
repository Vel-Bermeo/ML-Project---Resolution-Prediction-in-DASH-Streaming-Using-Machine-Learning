# ML-Project---Resolution-Prediction-in-DASH-Streaming-Using-Machine-Learning

# Machine Learning-Based Video Resolution Prediction in DASH Streaming

## Overview

This project explores the use of Machine Learning techniques to predict video resolution in HTTP Adaptive Streaming (HAS) environments. Using network and video segment metrics extracted from the Puffer dataset, the problem is formulated as a multiclass classification task where the objective is to predict the video resolution associated with a segment under varying network conditions.

The project includes data preprocessing, exploratory data analysis (EDA), feature selection, dimensionality reduction using t-SNE, model training, hyperparameter optimization with Optuna, and statistical comparison of machine learning models.

---

## Dataset

The experiments were conducted using data from the Puffer project developed by Stanford University.

Dataset source:
https://puffer.stanford.edu/data-description/

Selected features include:

* size
* ssim_index
* cwnd
* min_rtt
* rtt
* delivery_rate

Target variable:

* resolution

---

## Methodology

The following workflow was implemented:

1. Exploratory Data Analysis (EDA)
2. Class filtering and preprocessing
3. Train/Test Split
4. Feature Scaling
5. Feature Selection (Mutual Information)
6. t-SNE Visualization
7. Logistic Regression (Baseline)
8. Logistic Regression + Optuna
9. Learning Curve Analysis
10. Random Forest (Baseline)
11. Random Forest + Optuna
12. Learning Curve Analysis
13. Statistical Comparison using Wilcoxon Test

---

## Models Evaluated

### Logistic Regression

* Baseline model
* Hyperparameter optimization using Optuna
* Class imbalance handled with `class_weight="balanced"`

### Random Forest

* Baseline model
* Hyperparameter optimization using Optuna
* Feature selection integrated into Scikit-Learn Pipelines

---

## Results

| Model                        | Accuracy | F1 Macro |
| ---------------------------- | -------: | -------: |
| Logistic Regression          |   0.7226 |   0.4933 |
| Logistic Regression + Optuna |   0.7226 |   0.4929 |
| Random Forest                |   0.9115 |   0.7365 |
| Random Forest + Optuna       |   0.8944 |   0.7549 |

The optimized Random Forest model achieved the best overall performance, particularly improving the classification of minority classes.

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* Optuna
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Repository Structure

```text
├── data/
├── notebooks/
│   └── Proyecto_ML.ipynb
├── figures/
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/video-resolution-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## References

* Pedregosa et al. (2011). Scikit-Learn: Machine Learning in Python.
* Breiman (2001). Random Forests.
* Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework.
* van der Maaten & Hinton (2008). Visualizing Data using t-SNE.
* Puffer Project: https://puffer.stanford.edu

---

## Author

Evelyn N. Bermeo

Master's Program in Data Science

Universidad San Francisco de Quito (USFQ)
