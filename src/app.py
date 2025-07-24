from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pickle
import logging


# Prometheus
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST


# Logging
logging.basicConfig(filename="logs/prediction.log", level=logging.INFO)


# Load model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)


app = FastAPI()


# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Prometheus metrics
PREDICTION_COUNTER = Counter("prediction_requests_total", "Total prediction requests made")


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
