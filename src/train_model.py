"""
train_model.py  –  Agricultural Insurance Claim Prediction
Binary Classification: Logistic Regression vs XGBoost
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load & Clean
# ─────────────────────────────────────────────
df = pd.read_csv("data/agriculture_data.csv")
print(f"Shape: {df.shape}")
print(df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ─────────────────────────────────────────────
# 2. EDA Plots
# ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis – Agricultural Insurance Claims", fontsize=16, fontweight="bold")

# Target distribution
axes[0,0].pie(df["Insurance_Claim"].value_counts(), labels=["No Claim","Claim"],
              colors=COLORS, autopct="%1.1f%%", startangle=90)
axes[0,0].set_title("Target Distribution")

# Rainfall Deviation by Claim
df.boxplot(column="Rainfall_Deviation_Pct", by="Insurance_Claim", ax=axes[0,1],
           patch_artist=True)
axes[0,1].set_title("Rainfall Deviation vs Claim")
axes[0,1].set_xlabel("Insurance Claim"); axes[0,1].set_ylabel("Deviation (%)")

# Yield by Claim
sns.histplot(data=df, x="Yield_Tonnes_Per_Ha", hue="Insurance_Claim",
             palette={0:"#2ecc71",1:"#e74c3c"}, alpha=0.7, ax=axes[0,2], bins=30)
axes[0,2].set_title("Yield Distribution by Claim")

# Claim rate by Crop
crop_claim = df.groupby("Crop")["Insurance_Claim"].mean().sort_values(ascending=False)
crop_claim.plot(kind="bar", ax=axes[1,0], color="#3498db", edgecolor="black")
axes[1,0].set_title("Claim Rate by Crop"); axes[1,0].tick_params(axis="x", rotation=45)

# Claim rate by Soil
soil_claim = df.groupby("Soil_Type")["Insurance_Claim"].mean().sort_values(ascending=False)
soil_claim.plot(kind="bar", ax=axes[1,1], color="#9b59b6", edgecolor="black")
axes[1,1].set_title("Claim Rate by Soil Type")

# Correlation heatmap
num_cols = ["Area_Hectares","Production_Tonnes","Yield_Tonnes_Per_Ha",
            "Rainfall_Deviation_Pct","Soil_Quality_Score","Insurance_Claim"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1,2])
axes[1,2].set_title("Feature Correlations")

plt.tight_layout()
plt.savefig("reports/eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("EDA saved → reports/eda_plots.png")

# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────
df["Drought_Flag"]   = (df["Rainfall_Deviation_Pct"] < -20).astype(int)
df["Flood_Flag"]     = (df["Rainfall_Deviation_Pct"] >  30).astype(int)
df["Low_Yield_Flag"] = (df["Yield_Tonnes_Per_Ha"] < df["Yield_Tonnes_Per_Ha"].quantile(0.25)).astype(int)
df["Area_Log"]       = np.log1p(df["Area_Hectares"])
df["Prod_Log"]       = np.log1p(df["Production_Tonnes"])

cat_cols = ["State","Crop","Season","Soil_Type"]
le = LabelEncoder()
for c in cat_cols:
    df[c+"_Enc"] = le.fit_transform(df[c])

FEATURES = ["Rainfall_Deviation_Pct","Soil_Quality_Score","Yield_Tonnes_Per_Ha",
            "Area_Log","Prod_Log","Temperature_C","Fertiliser_Kg_Ha","Irrigation",
            "Drought_Flag","Flood_Flag","Low_Yield_Flag",
            "Crop_Enc","Season_Enc","Soil_Type_Enc","State_Enc"]

X = df[FEATURES]
y = df["Insurance_Claim"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(FEATURES, "models/features.pkl")

# ─────────────────────────────────────────────
# 4. Model Training
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -- Logistic Regression --
print("\n[1/2] Tuning Logistic Regression …")
lr_params = {"C":[0.01,0.1,1,10], "solver":["lbfgs","liblinear"],
             "class_weight":[None,"balanced"]}
lr_gs = GridSearchCV(LogisticRegression(max_iter=1000), lr_params,
                     cv=cv, scoring="recall", n_jobs=-1)
lr_gs.fit(X_train_s, y_train)
best_lr = lr_gs.best_estimator_
print("  Best params:", lr_gs.best_params_)

# -- XGBoost --
print("[2/2] Tuning XGBoost …")
xgb_params = {"n_estimators":[100,200], "max_depth":[3,5],
              "learning_rate":[0.05,0.1], "scale_pos_weight":[1,2]}
xgb_gs = GridSearchCV(XGBClassifier(use_label_encoder=False,
                                    eval_metric="logloss", random_state=42),
                      xgb_params, cv=cv, scoring="recall", n_jobs=-1)
xgb_gs.fit(X_train, y_train)
best_xgb = xgb_gs.best_estimator_
print("  Best params:", xgb_gs.best_params_)

# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
def evaluate(name, model, X_te, y_te, scaled=True):
    if not scaled:
        X_te = X_te
    pred  = model.predict(X_te)
    proba = model.predict_proba(X_te)[:,1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_te, pred),4),
        "Precision": round(precision_score(y_te, pred),4),
        "Recall":    round(recall_score(y_te, pred),4),
        "F1":        round(f1_score(y_te, pred),4),
        "ROC_AUC":   round(roc_auc_score(y_te, proba),4),
        "_pred":     pred, "_proba": proba
    }

lr_res  = evaluate("Logistic Regression", best_lr,  X_test_s, y_test)
xgb_res = evaluate("XGBoost",             best_xgb, X_test,   y_test, scaled=False)

results_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")}
                            for r in [lr_res, xgb_res]])
print("\n", results_df.to_string(index=False))

# ─────────────────────────────────────────────
# 6. Plots: Confusion Matrices + ROC
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

for ax, res, title in zip(axes[:2],
                           [lr_res, xgb_res],
                           ["Logistic Regression","XGBoost"]):
    cm = confusion_matrix(y_test, res["_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Claim","Claim"],
                yticklabels=["No Claim","Claim"])
    ax.set_title(f"Confusion Matrix – {title}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

for res, color, label in [(lr_res,"#3498db","Logistic Regression"),
                           (xgb_res,"#e74c3c","XGBoost")]:
    fpr, tpr, _ = roc_curve(y_test, res["_proba"])
    axes[2].plot(fpr, tpr, color=color,
                 label=f"{label} (AUC={res['ROC_AUC']:.3f})")
axes[2].plot([0,1],[0,1],"k--")
axes[2].set_title("ROC Curve"); axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
axes[2].legend()

plt.tight_layout()
plt.savefig("reports/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Evaluation plot saved → reports/model_evaluation.png")

# ─────────────────────────────────────────────
# 7. Feature Importance (XGBoost)
# ─────────────────────────────────────────────
fi = pd.Series(best_xgb.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8,6))
fi.plot(kind="barh", ax=ax, color="#e67e22")
ax.set_title("XGBoost Feature Importance"); ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 8. Select best model (Recall) & save
# ─────────────────────────────────────────────
best_name = results_df.loc[results_df["Recall"].idxmax(), "Model"]
print(f"\nBest model by Recall: {best_name}")

joblib.dump(best_lr,  "models/logistic_regression.pkl")
joblib.dump(best_xgb, "models/xgboost_model.pkl")

best_model  = best_xgb if "XGBoost" in best_name else best_lr
best_scaled = ("XGBoost" not in best_name)
joblib.dump({"model": best_model, "scaled": best_scaled, "name": best_name},
            "models/best_model.pkl")
print("Models saved to models/")

# ─────────────────────────────────────────────
# 9. Classification Reports
# ─────────────────────────────────────────────
for res in [lr_res, xgb_res]:
    print(f"\n── {res['Model']} ──")
    Xi = X_test_s if res["Model"]=="Logistic Regression" else X_test
    print(classification_report(y_test, res["_pred"],
                                 target_names=["No Claim","Claim"]))

results_df.to_csv("reports/model_comparison.csv", index=False)
print("\nAll done ✓")
