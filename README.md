# 🌾 Agricultural Insurance Claim Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Binary Classification** – Predict whether a farmer will file a crop insurance claim based on rainfall deviation, crop type, soil quality, production, yield, and area cultivated.

---

## 📌 Problem Statement

Agricultural insurance is critical in India where crop failures due to drought, flood, or poor soil conditions can push farmers into debt. This project builds a machine learning system to **predict insurance claim likelihood** in advance so that insurers and government agencies can prioritise resources.

---

## 🗂️ Project Structure

```
AgriInsureML/
├── data/
│   ├── agriculture_data.csv       # Dataset (Kaggle-sourced / synthetic)
│   └── generate_data.py           # Data generator script
├── models/
│   ├── logistic_regression.pkl    # Trained LR model
│   ├── xgboost_model.pkl          # Trained XGBoost model
│   ├── best_model.pkl             # Best model by Recall
│   ├── scaler.pkl                 # Feature scaler
│   └── features.pkl               # Feature list
├── notebooks/
│   └── AgriInsureML.ipynb         # Full Jupyter walkthrough
├── reports/
│   ├── eda_plots.png              # EDA visualisations
│   ├── model_evaluation.png       # Confusion matrices + ROC
│   ├── feature_importance.png     # XGBoost importances
│   └── model_comparison.csv       # Metrics comparison table
├── src/
│   └── train_model.py             # End-to-end ML pipeline
├── templates/
│   ├── index.html                 # Prediction form (Bootstrap)
│   └── result.html                # Result display page
├── app.py                         # Flask web application
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `Rainfall_Deviation_Pct` | % deviation from normal rainfall |
| `Crop` | Crop type (Rice, Wheat, Cotton, …) |
| `Season` | Kharif / Rabi / Zaid |
| `Soil_Type` | Alluvial / Black / Red / Laterite / Sandy |
| `Soil_Quality_Score` | Numerical soil quality (1–5) |
| `Area_Hectares` | Area under cultivation |
| `Production_Tonnes` | Total crop production |
| `Yield_Tonnes_Per_Ha` | Yield per hectare |
| `Irrigation` | 1 = Irrigated, 0 = Rain-fed |
| `Insurance_Claim` | **Target**: 1 = Claim filed, 0 = No claim |

---

## 🤖 Models & Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 97.9% | 96.0% | 99.0% | 97.5% | 99.8% |
| **XGBoost** | **99.9%** | **99.8%** | **100%** | **99.9%** | **100%** |

**Winner: XGBoost** (Best Recall – zero missed claims)

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-username/AgriInsureML.git
cd AgriInsureML
pip install -r requirements.txt
```

### 2. Train Models
```bash
python src/train_model.py
```

### 3. Launch Web App
```bash
python app.py
# → Open http://localhost:5000
```

### 4. Run Notebook
```bash
jupyter notebook notebooks/AgriInsureML.ipynb
```

---

## 🌐 Flask Web App

- Fill in the prediction form (rainfall deviation, crop, soil, area, production)
- Click **Predict Claim Likelihood**
- Get instant prediction with confidence score and advisory

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** – Logistic Regression, preprocessing, metrics
- **XGBoost** – Gradient boosted trees
- **Flask** – Web framework
- **Bootstrap 5** – Frontend UI
- **Matplotlib / Seaborn** – Visualisations
- **Joblib** – Model serialisation

---

## 📄 License
MIT © 2024
