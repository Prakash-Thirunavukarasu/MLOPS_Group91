import mlflow
from mlflow.tracking import MlflowClient
import os


os.environ["MLFLOW_REGISTRY_URI"] = "file:model_registry"


def promote_best_model(experiment_name="Iris_Classification", 
                       registered_model_name="Iris_Classifier_Prod"):
    client = MlflowClient()


    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return
    

    # Get best run by accuracy
    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    if not best_runs:
        print("No runs found in experiment.")
        return
    best_run = best_runs[0]
    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics["accuracy"]
    print(f"Best run ID: {best_run_id} | Accuracy: {best_accuracy:.4f}")


    # Use run-based model URI (avoid copying artifacts)
    model_uri = f"runs:/{best_run_id}/model"
    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )
        print(f"Registered model '{registered_model_name}' version {model_version.version}")
    except Exception as e:
        print(f"Model registration failed: {e}")
        return
    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Production",
            version=model_version.version
        )
        print(f"Set alias 'Production' → version {model_version.version}")
    except Exception as e:
        print(f"Setting alias failed: {e}")


if __name__ == "__main__":
    promote_best_model()