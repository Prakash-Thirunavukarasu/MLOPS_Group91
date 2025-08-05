from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import pickle
import logging
import time
import os
from mlflow.tracking import MlflowClient
from typing import List
import pandas as pd
from prometheus_client import Counter, Histogram


os.environ["MLFLOW_REGISTRY_URI"] = "file:model_registry"


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

# FastAPI app
app = FastAPI()
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


# Input schema
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length must be between 0 and 10")
    sepal_width: float = Field(..., gt=0, description="Sepal width must be between 0 and 10")
    petal_length: float = Field(..., gt=0, description="Petal length must be between 0 and 10")
    petal_width: float = Field(..., gt=0, description="Petal width must be between 0 and 10")


# Prometheus metrics
REQUEST_COUNT = Counter("iris_requests_total", "Total Iris data requests")
REQUEST_LATENCY = Histogram("iris_request_duration_seconds", "Duration of Iris data request")
prediction_counter = Counter('iris_predictions_total', 'Total number of predictions made')
prediction_latency = Histogram('iris_prediction_latency_seconds', 'Time spent making predictions')
prediction_class_counter = Counter('iris_prediction_class_total', 'Predictions per class', ['class'])


iris_df = pd.DataFrame([
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
])


@app.get("/")
def read_root():
    return {"message": "FastAPI is working!.Welcome to the Iris API"}


@app.post("/predict")
def predict(input: IrisInput):
    REQUEST_COUNT.inc()
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    prediction = model.predict(data)[0]
    logging.info(f"Input: {input.dict()} | Prediction: {prediction}")
    return {"prediction": prediction}


@app.get("/iris", response_model=List[IrisInput])
def get_iris_data():
    REQUEST_COUNT.inc()
    start_time = time.time()
    time.sleep(0.1)
    duration = time.time() - start_time
    REQUEST_LATENCY.observe(duration)
    records = iris_df.to_dict(orient="records")
    return [IrisInput(**record) for record in records]
