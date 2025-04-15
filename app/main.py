from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Create the FastAPI app
app = FastAPI(title="ML Model API")

# Load the pre-trained model
model = joblib.load("app/model.pkl")

# Define the input data structure
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input to the format the model expects
    features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction
    return {"prediction": int(prediction[0])}