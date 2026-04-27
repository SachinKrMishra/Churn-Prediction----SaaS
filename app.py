import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from churnprediction.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()
pipeline = PredictionPipeline()


class ChurnRequest(BaseModel):
    Age: int
    Number_of_Subscriptions: int
    Avg_Usage_Hours_Per_Week: float
    App_Switch_Frequency: int
    Discount_Used: int
    Customer_Support_Interactions: int
    Tenure_Months: int
    Monthly_Total_Spend: float
    Income_Level: str
    Payment_Mode: str
    Device_Type: str


@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}


@app.post("/predict")
def predict(data: ChurnRequest):
    try:
        df = pd.DataFrame([data.dict()])

        prediction, probability, pos, neg, actions, base, shap_vals, feature_names = pipeline.predict(df)

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "top_positive": pos,
            "top_negative": neg,
            "actions": actions,
            "shap_values": shap_vals,
            "feature_names": feature_names
        }

    except Exception as e:
        return {"error": str(e)}