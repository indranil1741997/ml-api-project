from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

# Load model
model = joblib.load("app/model.pkl")

# Iris target names
target_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Input model
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(features)
    class_name = target_names[prediction[0]]
    return {"prediction": class_name}
