import pickle
import json
import pandas as pd
import streamlit as st
from datetime import datetime

with open('rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.json', 'r') as f:
    features = json.load(f)
    

st.title("Fraud Detection Prediction")
st.subheader("Input Transaction Details")

account_age_days = st.number_input("Account Age (days)", min_value=1, value=100)
amount = st.number_input("Transaction Amount", min_value=1, value=5000)
shipping_distance_km = st.number_input("Shipping Distance (km)", min_value=0, value=10)
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
transaction_date = st.date_input("Transaction Date", datetime.today())

# Extract time features
day_of_week = transaction_date.weekday()
day_of_month = transaction_date.day
month = transaction_date.month

# input df
input_df = pd.DataFrame([[account_age_days, amount, shipping_distance_km, hour, day_of_week, day_of_month, month]], columns=features)

input_scaled = scaler.transform(input_df)

# predict momen :D
pred_prob = rf.predict_proba(input_scaled)[0][1]
pred_label = rf.predict(input_scaled)[0]

# Display result
st.subheader("Prediction Result")
st.write(f"Fraud Probability: {pred_prob:.2f}")
st.write(f"Predicted Label: {'Fraud' if pred_label==1 else 'Not Fraud'}")

