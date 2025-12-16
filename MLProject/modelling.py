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
        # Cek file .npy terlebih dahulu
        x_train_path = os.path.join(data_path, "X_train.npy")
        x_test_path = os.path.join(data_path, "X_test.npy")
        y_train_path = os.path.join(data_path, "y_train.csv")
        y_test_path = os.path.join(data_path, "y_test.csv")
        
        # Periksa apakah file preprocessing ada
        if all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
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
        else:
            print("Preprocessed files not found, trying CSV...")
    
    # fallback: CSV
    # Cek apakah file CSV ada
    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully from CSV. Shape: {df.shape}")
        return df
    else:
        # Coba cari file CSV di dalam folder
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(data_path, csv_files[0])
            df = pd.read_csv(csv_path)
            print(f"Data loaded from {csv_path}. Shape: {df.shape}")
            return df
        else:
            raise FileNotFoundError(f"No data found at {data_path}")

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocessing data dan split train-test"""
    print("Preprocessing data...")
    
    # Cek nama kolom target
    # Coba beberapa nama kolom target yang umum
    target_columns = ['target', 'Target', 'TARGET', 'label', 'Label', 'LABEL', 'class', 'Class', 'CLASS']
    
    target_col = None
    for col in target_columns:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Asumsi kolom terakhir adalah target
        target_col = df.columns[-1]
        print(f"Target column not found, using last column: {target_col}")
    
    # Pisahkan fitur dan target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Training model"""
    print("Training model...")
    
    # Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
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
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, y_pred, cm

def save_model_artifacts(model, model_name, X_test, y_test, y_pred, accuracy):
    """Simpan model dan artifacts"""
    # Buat folder artifacts jika belum ada
    os.makedirs('artifacts', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"artifacts/{model_name}_{timestamp}.pkl"
    
    # Simpan model
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Simpan metrics
    metrics_filename = f"artifacts/metrics_{timestamp}.txt"
    with open(metrics_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Train samples: {len(X_test) * 4} (estimated)\n")
    
    print(f"Model saved to {model_filename}")
    print(f"Metrics saved to {metrics_filename}")
    return model_filename, metrics_filename

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train ML Model with MLflow')
    parser.add_argument('--data_path', type=str, default='./',
                       help='Path ke CSV atau folder hasil preprocessing')
    parser.add_argument('--model_name', type=str, default='model',
                       help='Model name for saving')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of estimators for RandomForest')
    parser.add_argument('--max_depth', type=int, default=10,
                       help='Max depth for RandomForest')
    
    args = parser.parse_args()
    
    # Setup MLflow - JANGAN set experiment di sini, biarkan dari command line
    # mlflow.set_experiment("ML_Training_CI")
    
    # Mulai run MLflow
    with mlflow.start_run():
        print("=" * 60)
        print("MLflow Training Started")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)
        
        # Log parameters
        mlflow.log_params({
            "data_path": args.data_path,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "model_name": args.model_name
        })
        
        try:
            # Load data
            loaded = load_data(args.data_path)
            
            # Preprocess
            if isinstance(loaded, tuple) and len(loaded) == 4:
                X_train, X_test, y_train, y_test = loaded
                print("Using preprocessed data")
            else:
                df = loaded
                X_train, X_test, y_train, y_test = preprocess_data(
                    df, args.test_size, args.random_state
                )
            
            # Train model
            model = train_model(X_train, y_train, args.n_estimators, args.max_depth)
            
            # Evaluate
            accuracy, y_pred, cm = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"{args.model_name}_v1"
            )
            
            # Save artifacts
            model_path, metrics_path = save_model_artifacts(
                model, args.model_name, X_test, y_test, y_pred, accuracy
            )
            
            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact("artifacts/")
            
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Model saved: {model_path}")
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            print(f"MLflow Experiment ID: {mlflow.active_run().info.experiment_id}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nERROR during training: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

if __name__ == "__main__":
    main()