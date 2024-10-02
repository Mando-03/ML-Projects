from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = FastAPI()

df = pd.read_csv('House Prices.csv')
df = df.drop(columns='ID')
X = df.drop(columns='medv')
scaler = StandardScaler()
scaler.fit(X)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('stacking_regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the request body using Pydantic
class HouseData(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    black: float
    lstat: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: HouseData):
    try:
        # Convert the input data to a numpy array
        input_data = np.array([[
            data.crim, data.zn, data.indus, data.chas, data.nox, data.rm,
            data.age, data.dis, data.rad, data.tax, data.ptratio, data.black, data.lstat
        ]])

        
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

        # Return the prediction
        return {"medv": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Welcome to the House Price Prediction API"}
