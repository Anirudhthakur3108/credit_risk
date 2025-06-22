import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# Load model and scaler
def load_gz_model(filename):
    with gzip.open(filename, 'rb') as f:
        return joblib.load(f)

model = load_gz_model("rf_credit_model.pkl.gz")
scaler = load_gz_model("scaler.pkl.gz")

st.title("Credit Risk Prediction App")

def user_input_features():
    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual Income", 0, 1000000, 50000)
    person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    person_emp_length = st.number_input("Employment Length (Years)", 0, 100, 5)
    loan_intent = st.selectbox("Loan Intent", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt = st.number_input("Loan Amount", 500, 40000, 10000)
    loan_int_rate = st.number_input("Interest Rate", 5.0, 30.0, 15.0)
    loan_percent_income = st.number_input("Loan Percent of Income", 0.0, 1.0, 0.2)
    cb_person_default_on_file = st.selectbox("Previously Defaulted?", ['Y', 'N'])
    cb_person_cred_hist_length = st.number_input("Credit History Length", 1, 30, 5)

    data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'].index(person_home_ownership),
        'person_emp_length': person_emp_length,
        'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'].index(loan_intent),
        'loan_grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G'].index(loan_grade),
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': ['Y', 'N'].index(cb_person_default_on_file),
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predict
if st.button("Predict Credit Risk"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = "Default" if prediction[0] == 1 else "No Default"
    st.success(f"Prediction: {result}")