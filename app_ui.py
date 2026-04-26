import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn probability")

# Inputs
Age = st.number_input("Age", min_value=10, max_value=100, value=30)
Number_of_Subscriptions = st.number_input("Number of Subscriptions", 0, 10, 2)
Avg_Usage_Hours_Per_Week = st.number_input("Avg Usage Hours/Week", 0.0, 100.0, 10.0)
App_Switch_Frequency = st.number_input("App Switch Frequency", 0, 10, 1)
Discount_Used = st.number_input("Discount Used", 0, 10, 1)
Customer_Support_Interactions = st.number_input("Support Interactions", 0, 20, 0)
Tenure_Months = st.number_input("Tenure (Months)", 0, 60, 12)
Monthly_Total_Spend = st.number_input("Monthly Spend", 0.0, 10000.0, 500.0)

Income_Level = st.selectbox("Income Level", ["Low", "Medium", "High"])
Payment_Mode = st.selectbox("Payment Mode", ["Credit Card", "Debit Card", "UPI", "Net Banking"])
Device_Type = st.selectbox("Device Type", ["Mobile", "Tablet", "Desktop"])

# Button
if st.button("Predict Churn"):
    input_data = {
        "Age": Age,
        "Number_of_Subscriptions": Number_of_Subscriptions,
        "Avg_Usage_Hours_Per_Week": Avg_Usage_Hours_Per_Week,
        "App_Switch_Frequency": App_Switch_Frequency,
        "Discount_Used": Discount_Used,
        "Customer_Support_Interactions": Customer_Support_Interactions,
        "Tenure_Months": Tenure_Months,
        "Monthly_Total_Spend": Monthly_Total_Spend,
        "Income_Level": Income_Level,
        "Payment_Mode": Payment_Mode,
        "Device_Type": Device_Type
    }

    try:
        response = requests.post("http://api:8000/predict")

        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: {'Churn' if result['prediction'] == 1 else 'No Churn'}")
            st.info(f"Probability: {round(result['probability'], 2)}")

        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Error: {e}")