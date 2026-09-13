from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel, Field

app = FastAPI()

model_pipeline = joblib.load("models/transit_model.pkl")

class TripFeatures(BaseModel):
    temperature_C: float = Field(..., json_schema_extra={"example": 22.5})
    traffic_congestion_index: float = Field(..., json_schema_extra={"example": 7.5})
    hour: int = Field(..., ge=0, le=23, json_schema_extra={"example": 17})
    day_of_week: int = Field(..., ge=0, le=6, json_schema_extra={"example": 2})
    event_attendance_est: float = Field(..., json_schema_extra={"example": 5000.0})
    transport_type: str = Field(..., json_schema_extra={"example": "Bus"})
    weather_condition: str = Field(..., json_schema_extra={"example": "Rain"})
    event_type: str = Field(..., json_schema_extra={"example": "Concert"})
    route_id: str = Field(..., json_schema_extra={"example": "R01"})

@app.post("/predict")
def predict(features: TripFeatures):
    try:
        input_df = pd.DataFrame([features.dict()])
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[:, 1][0]
        return {
            "delayed_prediction": int(prediction),
            "delay_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
