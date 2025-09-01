import os, json, joblib, datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ============================
# Paths
# ============================
BASE_DIR = "parkinsons_final"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "parkinsons.csv")

# ============================
# Load dataset
# ============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}. Please put parkinsons.csv there.")

df = pd.read_csv(DATA_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])

X = df.drop("status", axis=1)
y = df["status"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Define models
# ============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(probability=True, kernel="rbf"),
    "MLP (Sklearn)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
}

# ============================
# Train & evaluate
# ============================
leaderboard = {}
best_auc = -1
best_model = None
best_name = None

for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    y_prob_train = pipe.predict_proba(X_train)[:,1]
    y_prob_test = pipe.predict_proba(X_test)[:,1]

    # Metrics
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

    # Save best model
    if metrics_test["roc_auc"] > best_auc:
        best_auc = metrics_test["roc_auc"]
        best_model = pipe
        best_name = name
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{name}_model.joblib"))

# ============================
# Save artifacts
# ============================
# Best model
best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
joblib.dump(best_model, best_model_path)

# Metrics of best model
metrics_path = os.path.join(ASSETS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(leaderboard[best_name], f, indent=2)

# Leaderboard of all models
leaderboard_path = os.path.join(ASSETS_DIR, "leaderboard.json")
with open(leaderboard_path, "w") as f:
    json.dump(leaderboard, f, indent=2)

# Training log
log_path = os.path.join(ASSETS_DIR, "training_log.csv")
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = {"timestamp": now, "best_model": best_name, "roc_auc": best_auc}
log_df = pd.DataFrame([log_entry])
if os.path.exists(log_path):
    log_df.to_csv(log_path, mode="a", header=False, index=False)
else:
    log_df.to_csv(log_path, index=False)

print(f"✅ Training complete. Best model: {best_name} (ROC-AUC={best_auc:.3f})")
