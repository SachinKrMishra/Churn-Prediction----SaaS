# 🚀 Customer Churn Prediction System (End-to-End ML Project)

An end-to-end **Machine Learning system** to predict customer churn using a production-ready pipeline, API, UI, and Dockerized deployment.

---

## 📌 Project Overview

This project predicts whether a customer will **cancel their subscription in the next 3 months** using behavioral and transactional features.

It is built with a **complete ML lifecycle**:

* Data Ingestion
* Data Validation (Schema + Drift Detection)
* Data Transformation (Feature Engineering + Pipelines)
* Model Training (Hyperparameter Tuning + MLflow Tracking)
* Prediction Pipeline (Explainability (SHAP))
* FastAPI Backend (Model serving)
* Streamlit Frontend (User interface)
* Dockerized Deployment (Multi-container setup)

---

## 🧠 Problem Statement

Subscription-based platforms face revenue loss due to churn.
This system helps proactively identify customers likely to churn.

---

## 🏗️ System Architecture

```
User → Streamlit UI → FastAPI → ML Model → Prediction → UI
```

### Pipeline Flow:

```
Raw Data → Ingestion → Validation → Transformation → Model Training → Saved Model
```

---

## ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Random Forest**
* **XGBoost**
* **MLflow**
* **SHAP**
* **FastAPI**
* **Streamlit**
* **Docker & Docker Compose**

---

## 🚀 Features

✔ Modular ML pipeline (industry-style)
✔ Data validation with schema + drift detection
✔ Feature engineering using ColumnTransformer
✔ Hyperparameter tuning with GridSearchCV
✔ Experiment tracking using MLflow
✔ Explainability using SHAP
✔ REST API using FastAPI
✔ Interactive UI using Streamlit
✔ Dockerized multi-service deployment

---

## 📊 Model Details

Models used:

* Logistic Regression
* Random Forest
* XGBoost

Evaluation Metrics:

* F1 Score
* Precision
* Recall
* ROC-AUC

Best model selected based on **F1 Score**

---

## 🔮 API Usage

### Endpoint:

```
POST /predict
```

### Sample Request:

```json
{
  "Age": 30,
  "Number_of_Subscriptions": 3,
  "Avg_Usage_Hours_Per_Week": 12.5,
  "App_Switch_Frequency": 2,
  "Discount_Used": 1,
  "Customer_Support_Interactions": 0,
  "Tenure_Months": 12,
  "Monthly_Total_Spend": 500,
  "Income_Level": "Medium",
  "Payment_Mode": "UPI",
  "Device_Type": "Mobile"
}
```

### Response:

```json
{
  "prediction": 0,
  "probability": 0.21
}
```

---

## 🖥️ Run Locally

### 1. Train Model

```bash
python main.py
```

### 2. Run API

```bash
uvicorn app:app --reload
```

### 3. Run UI

```bash
streamlit run app_ui.py
```

---

## 🐳 Run with Docker

### Build & Run

```bash
docker-compose up --build
```

### Access

* API → http://localhost:8000/docs
* UI → http://localhost:8501

---

## ⚠️ Notes

* Dataset is **excluded from Docker image** using `.dockerignore`
* Only trained model artifacts are included for inference
* System follows **stateless container design**

---

