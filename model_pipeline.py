import os
import json
import joblib
import datetime
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "parkinsons.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Auto-download dataset if missing
if not os.path.exists(DATA_PATH):
    print("⚠️ Dataset not found locally – downloading from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df_raw = pd.read_csv(url, header=0)
    df_raw.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset downloaded and saved to {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])

X = df.drop("status", axis=1)
y = df["status"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models to train
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
}

leaderboard = {}
best_auc = -1
best_model = None
best_name = None

for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    y_prob_train = pipe.predict_proba(X_train)[:,1]
    y_prob_test = pipe.predict_proba(X_test)[:,1]
    metrics_train = {
        "accuracy": accuracy_score(y_train, y_pred_train),
        "precision": precision_score(y_train, y_pred_train),
        "recall": recall_score(y_train, y_pred_train),
        "f1": f1_score(y_train, y_pred_train),
        "roc_auc": roc_auc_score(y_train, y_prob_train),
    }
    metrics_test = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
        "roc_auc": roc_auc_score(y_test, y_prob_test),
    }
    leaderboard[name] = {"train": metrics_train, "test": metrics_test}
    if metrics_test["roc_auc"] > best_auc:
        best_auc = metrics_test["roc_auc"]
        best_model = pipe
        best_name = name
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{name}_model.joblib"))

best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
joblib.dump(best_model, best_model_path)

metrics_path = os.path.join(ASSETS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(leaderboard[best_name], f, indent=2)

leaderboard_path = os.path.join(ASSETS_DIR, "leaderboard.json")
with open(leaderboard_path, "w") as f:
    json.dump(leaderboard, f, indent=2)

log_path = os.path.join(ASSETS_DIR, "training_log.csv")
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = {"timestamp": now, "best_model": best_name, "roc_auc": best_auc}
log_df = pd.DataFrame([log_entry])
if os.path.exists(log_path):
    log_df.to_csv(log_path, mode="a", header=False, index=False)
else:
    log_df.to_csv(log_path, index=False)

def compute_shap_values(model, X_sample):
    try:
        explainer = shap.Explainer(model.named_steps["clf"], X_sample)
        shap_values = explainer(X_sample)
        return shap_values
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None
