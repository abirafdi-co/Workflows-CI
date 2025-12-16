import os
import mlflow

def main():
    args = parse_args()

    # Kalau dijalankan via `mlflow run`, MLflow Project sudah start run.
    running_as_project = os.getenv("MLFLOW_RUN_ID") is not None

    if not running_as_project:
        mlflow.start_run()

    try:
        # logging tetap sama
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

        print("Training completed successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Model saved: {model_path}")

    finally:
        if not running_as_project:
            mlflow.end_run()
