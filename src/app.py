# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import logging


# Logging
logging.basicConfig(filename="logs/prediction.log", level=logging.INFO)


# Load model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)


app = FastAPI()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(input: IrisInput):
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    prediction = model.predict(data)[0]
    logging.info(f"Input: {input.dict()} | Prediction: {prediction}")
    return {"prediction": prediction}
