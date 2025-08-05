import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import preprocess
import promote_model
import warnings
import shutil
warnings.filterwarnings('ignore')

# Set up clean MLflow tracking and registry paths
mlflow.set_tracking_uri("file:mlruns")
os.environ["MLFLOW_REGISTRY_URI"] = "file:model_registry"
mlflow.set_experiment("Iris_Classification")


def train_models(data_folder="data/processed"):
    """
    Loads processed Iris data, trains multiple classification models,
    and tracks experiments with MLflow.
    """
    # Preprocess the data
    preprocess.preprocess_data()

    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

    X_train = train_df.drop("species", axis=1)
    y_train = train_df["species"]
    X_test = test_df.drop("species", axis=1)
    y_test = test_df["species"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100)
    }

    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")

        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model_type", model_name)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.set_tag("model_name", model_name)

            # Example input (e.g., your test dataset)
            input_example = X_test.iloc[:5]  # or use a representative sample
            # Infer signature
            signature = infer_signature(X_train, model.predict(X_train))
            # Log model under run, but don't auto-register
            mlflow.sklearn.log_model(model, name="model", input_example=input_example, signature=signature)

            print(f"Accuracy: {accuracy:.4f} | Model logged under run ID: {run.info.run_id}")

    print("All models trained and logged.")
    print("Run 'mlflow ui' to view results.")

    # Promote the best model
    promote_model.promote_best_model()

    # Removing the trash
    shutil.rmtree("mlruns/.trash", ignore_errors=True)


if __name__ == "__main__":
    train_models()
