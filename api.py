from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define the model for device specifications
class DeviceSpecs(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

# Load the trained model
model = joblib.load('device_price_predictor.joblib')

@app.post("/predict")
def predict(device: DeviceSpecs):
    # Prepare the input features
    features = [
        device.battery_power,
        device.blue,
        device.clock_speed,
        device.dual_sim,
        device.fc,
        device.four_g,
        device.int_memory,
        device.m_dep,
        device.mobile_wt,
        device.n_cores,
        device.pc,
        device.px_height,
        device.px_width,
        device.ram,
        device.sc_h,
        device.sc_w,
        device.talk_time,
        device.three_g,
        device.touch_screen,
        device.wifi,
    ]

    # Standardize the input features
    features_scaled = scaler.transform([features])

    # Get the predicted price range
    prediction = model.predict(features_scaled)

    return {"predicted_price_range": int(prediction[0])}
