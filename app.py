import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from churnprediction.pipeline.prediction_pipeline import PredictionPipeline

# Initialize FastAPI app
app = FastAPI()

# Load pipeline ONCE (important for performance)
pipeline = PredictionPipeline()


# Input schema
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


# Home route
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# Prediction route
@app.post("/predict")
def predict(data: ChurnRequest):
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Prediction
        pred, prob = pipeline.predict(df)

        return {
            "prediction": int(pred[0]),
            "probability": float(prob[0])
        }

    except Exception as e:
        return {"error": str(e)}