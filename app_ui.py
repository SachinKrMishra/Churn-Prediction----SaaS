import streamlit as st
import requests

st.set_page_config(layout="wide")

# ---------------- FIXED CSS (READABILITY FIRST) ----------------
st.markdown("""
<style>

html, body {
    background: #0b0f14;
    color: #ffffff;
}

/* Header */
.header {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtext {
    color: #b0b8c1;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

/* Result */
.result-good {
    color: #22c55e;
    font-size: 22px;
    font-weight: 600;
}

.result-bad {
    color: #ef4444;
    font-size: 22px;
    font-weight: 600;
}

/* Probability */
.prob {
    font-size: 38px;
    font-weight: 700;
    text-align: right;
}

/* Section title */
.section-title {
    font-size: 18px;
    margin-bottom: 10px;
    color: #d1d5db;
}

/* Reasons (HIGH CONTRAST FIX) */
.reason {
    background: #2a2e39;
    padding: 14px;
    border-radius: 8px;
    margin-bottom: 10px;
    color: #ffffff;
    font-size: 15px;
    line-height: 1.4;
}

/* Left indicators */
.reason.pos {
    border-left: 5px solid #ef4444;
}

.reason.neg {
    border-left: 5px solid #22c55e;
}

/* Actions */
.action {
    background: #263238;
    padding: 14px;
    border-radius: 8px;
    margin-bottom: 10px;
    color: #ffffff;
    font-size: 15px;
}

/* Button */
button[kind="primary"] {
    width: 100%;
    height: 48px;
    font-size: 16px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='header'>📊 Customer Churn Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Predict churn risk and understand key drivers</div>", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 10, 100, 30)
    Number_of_Subscriptions = st.number_input("Subscriptions", 0, 10, 2)
    Avg_Usage_Hours_Per_Week = st.number_input("Usage Hours", 0.0, 100.0, 10.0)

with col2:
    App_Switch_Frequency = st.number_input("Switch Frequency", 0, 50, 5)
    Discount_Used = st.number_input("Discount Used", 0, 20, 1)
    Customer_Support_Interactions = st.number_input("Support Interactions", 0, 20, 2)

with col3:
    Tenure_Months = st.number_input("Tenure (Months)", 0, 60, 12)
    Monthly_Total_Spend = st.number_input("Monthly Spend", 0.0, 10000.0, 500.0)

Income_Level = st.selectbox("Income Level", ["Low", "Medium", "High"])
Payment_Mode = st.selectbox("Payment Mode", ["UPI", "Credit Card", "Debit Card"])
Device_Type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])

# ---------------- ACTION ----------------
if st.button("Analyze Customer"):

    payload = {
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
        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error("API Error")
            st.text(response.text)

        else:
            res = response.json()

            prediction = res["prediction"]
            prob = res["probability"]
            pos = res["top_positive"]
            neg = res["top_negative"]
            actions = res["actions"]

            # -------- RESULT --------
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            col1, col2 = st.columns([3,1])

            with col1:
                if prediction == 1:
                    st.markdown("<div class='result-bad'>High Churn Risk</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-good'>Low Churn Risk</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"<div class='prob'>{round(prob*100,1)}%</div>", unsafe_allow_html=True)

            st.progress(prob)

            st.markdown("</div>", unsafe_allow_html=True)

            # -------- DRIVERS --------
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Key Drivers</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Risk Drivers**")
                for r in pos:
                    st.markdown(
                        f"<div class='reason pos'>{r['explanation']} → +{round(r['impact'],3)}</div>",
                        unsafe_allow_html=True
                    )

            with col2:
                st.markdown("**Protective Factors**")
                for r in neg:
                    st.markdown(
                        f"<div class='reason neg'>{r['explanation']} → {round(r['impact'],3)}</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # -------- ACTIONS --------
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Recommended Actions</div>", unsafe_allow_html=True)

            for a in actions:
                st.markdown(f"<div class='action'>• {a}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")