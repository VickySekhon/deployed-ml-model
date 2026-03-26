import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

filename = "./MNIST_predictor.pkl"
with open(filename, "rb") as f:
    model = pickle.load(f)

""" 
Pydantic model:
1) Enforces types at runtime
2) Provides detailed error if input doesn't match
3) Generates OpenAPI documentation for an API with the input type included
"""
class API_Input(BaseModel):
    pixels: list[float]

def normalize_values(df):
    return df / 255

app = FastAPI()

@app.post("/get_prediction")
async def root(sample: API_Input):
    if len(sample.pixels) != 784:
        return {"Error": f"Invalid input (should be 784 pixel values)"}
    pixels = pd.DataFrame([sample.pixels])
    X_test = normalize_values(pixels)
    prediction = model.predict(X_test)
    return {"Prediction": f"{int(prediction[0])}"}
