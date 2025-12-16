import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import argparse
import pickle
import os
from datetime import datetime

def load_data(data_path: str):
    """Load dataset.

    Dukungan:
    1) CSV (kolom target bernama 'target')
    2) Folder hasil preprocessing berisi:
       - X_train.npy, X_test.npy
       - y_train.csv, y_test.csv
    """
    print(f"Loading data from {data_path}...")

    if os.path.isdir(data_path):
        x_train_path = os.path.join(data_path, "X_train.npy")
        x_test_path = os.path.join(data_path, "X_test.npy")
        y_train_path = os.path.join(data_path, "y_train.csv")
        y_test_path = os.path.join(data_path, "y_test.csv")

        X_train = np.load(x_train_path)
        X_test = np.load(x_test_path)
        y_train = pd.read_csv(y_train_path).squeeze("columns")
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        print(
            f"Loaded preprocessed arrays. "
            f"X_train={X_train.shape}, X_test={X_test.shape}, "
            f"y_train={y_train.shape}, y_test={y_test.shape}"
        )
        return (X_train, X_test, y_train, y_test)

    # fallback: CSV
    df = pd.read_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocessing data dan split train-test"""
    print("Preprocessing data...")
    
    # Pisahkan fitur dan target
    # Sesuaikan dengan nama kolom target Anda
    X = df.drop('target', axis=1)  # Ganti 'target' dengan nama kolom target Anda
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Training model"""
    print("Training model...")
    
    # Model - sesuaikan dengan model Anda
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluasi model"""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred

def save_model_artifacts(model, model_name):
    """Simpan model dan artifacts"""
    # Buat folder artifacts jika belum ada
    os.makedirs('artifacts', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"artifacts/{model_name}_{timestamp}.pkl"
    
    # Simpan model
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_filename}")
    return model_filename

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train ML Model with MLflow')
    parser.add_argument('--data_path', type=str, default='telco_preprocessing',
                        help='Path ke CSV (kolom target=target) atau folder hasil preprocessing')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name for saving')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # MLflow tracking
    mlflow.set_experiment("ML_Training_CI")
    
    project_run_id = os.getenv("MLFLOW_RUN_ID")
    
    if mlflow.active_run() is None:
        if project_run_id:
            mlflow.start_run(run_id=project_run_id)
        else:
            mlflow.start_run()

    try:
    # Log parameters
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        
        loaded = load_data(args.data_path)
        
        if isinstance(loaded, tuple) and len(loaded) == 4:
            X_train, X_test, y_train, y_test = loaded
        else:
            df = loaded
            X_train, X_test, y_train, y_test = preprocess_data(
            df, args.test_size, args.random_state
        )
        model = train_model(X_train, y_train)
        accuracy, y_pred = evaluate_model(model, X_test, y_test)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", int(len(X_train)))
        mlflow.log_metric("test_samples", int(len(X_test)))
        
        mlflow.sklearn.log_model(model, "model")
        
        model_path = save_model_artifacts(model, args.model_name)
        mlflow.log_artifact(model_path)
        
        print(f"\n{'='*50}")
        print("Training completed successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved: {model_path}")
        print(f"{'='*50}\n")
    finally:
        if not project_run_id:
            mlflow.end_run()

if __name__ == "__main__":
    main()