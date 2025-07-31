from fastapi import FastAPI, Response
from pydantic import BaseModel
import pickle
import logging
import os
from mlflow.tracking import MlflowClient
import os

os.environ["MLFLOW_REGISTRY_URI"] = "file:model_registry"


# Prometheus
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST


# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/prediction.log",
                    level=logging.INFO)


# Latest model path
client = MlflowClient()
mv = client.get_model_version_by_alias("Iris_Classifier_Prod", "Production")
experiment = client.get_experiment_by_name("Iris_Classification")
model_path = 'mlruns/'+experiment.experiment_id+'/models/'+mv.source.split('/')[1]+'/artifacts/'+'model.pkl'


# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)


app = FastAPI()


# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Prometheus metrics
PREDICTION_COUNTER = Counter("prediction_requests_total",
                             "Total prediction requests made")


@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}


@app.post("/predict")
def predict(input: IrisInput):
    PREDICTION_COUNTER.inc()
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    prediction = model.predict(data)[0]
    logging.info(f"Input: {input.dict()} | Prediction: {prediction}")
    return {"prediction": prediction}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)