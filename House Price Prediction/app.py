from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

class PredictionInput(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: float
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    black: float
    lstat: float

scaler = joblib.load('scaler.pkl')
model = joblib.load('best_stacking_regressor.pkl')

app = FastAPI()

@app.post("/predict")
async def predict(input_data: PredictionInput):
    #convert input data to numpy array
    data = np.array([[input_data.crim, input_data.zn, input_data.indus, input_data.chas,
                    input_data.nox, input_data.rm, input_data.age, input_data.dis,
                    input_data.rad, input_data.tax, input_data.ptratio, input_data.black,
                    input_data.lstat]])

    #scale the input data
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return {"medv": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
