import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")

# Inputs (simple)
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges")

gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Convert input to dataframe
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "gender": [gender],
    "Contract": [contract]
})

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Customer will Churn ❌")
    else:
        st.success("Customer will Stay ✅")