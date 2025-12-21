import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.linear_model import LogisticRegression


dagshub.init(
    repo_owner="abirafdi-co",
    repo_name="Membangun_model",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/abirafdi-co/Membangun_model.mlflow"
)

DATA_DIR = "telco_preprocessing"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).iloc[:, 0]
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).iloc[:, 0]

mlflow.set_experiment("telco_churn_ci")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="ci_retrain"):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
