# ml_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")  # ML Model loading

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.feature1, data.feature2]])
    return {"prediction": prediction[0]}
