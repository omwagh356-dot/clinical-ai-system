pip install --upgrade pip
import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Clinical AI Portal", page_icon="🏥")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_assets()
except Exception as e:
    st.error("Model files not found. Please upload model.h5, scaler.pkl, and label_encoder.pkl.")

def send_to_doctor(receiver_email, data):
    # --- NEW: Define the email content here ---
    msg = EmailMessage()
    msg['Subject'] = f"Urgent Health Report: {data['name']}"
    msg['From'] = st.secrets["EMAIL_USER"]
    msg['To'] = receiver_email
    
    msg.set_content(f"""
    Patient Clinical Report
    -----------------------
    Name: {data['name']}
    Age: {data['age']}
    
    Diagnosis: {data['disease']}
    Confidence: {data['prob']}%
    Risk Level: {data['risk']}
    
    Vitals Summary:
    HR: {data['hr']} | SpO2: {data['spo2']}% | Temp: {data['temp']}°C
    BP: {data['bps']}/{data['bpd']} | Glucose: {data['gluc']}
    """)
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"]) 
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False

# --- UI LAYOUT ---
st.title("🏥 Clinical AI Diagnostic System")
st.markdown("Enter patient vitals below to generate an AI-driven risk assessment.")

with st.form("diagnostic_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name")
        doc_email = st.text_input("Doctor Email")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        hr = st.number_input("Heart Rate (BPM)", value=72)
        bps = st.number_input("BP Systolic", value=120)
    with col2:
        spo2 = st.number_input("SpO2 %", value=98)
        temp = st.number_input("Temperature °C", value=37.0)
        gluc = st.number_input("Glucose mg/dL", value=95)
        resp = st.number_input("Respiratory Rate", value=16)
        bpd = st.number_input("BP Diastolic", value=80)

    submitted = st.form_submit_button("RUN DIAGNOSTIC")

if submitted:
    # 1. Prepare Data
    inputs = np.array([[age, hr, bps, bpd, spo2, temp, 200, gluc, resp]]) 

    # 2. Predict
    scaled_data = scaler.transform(inputs)
    prediction = model.predict(scaled_data)
    idx = np.argmax(prediction)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = np.max(prediction) * 100

    # 3. Triage Logic
    risk = "Low"
    color = "blue"
    if spo2 < 90 or bps >= 180 or temp > 39.5:
        risk = "High"
        color = "red"

    # 4. Display Results
    st.markdown(f"### Result: :{color}[{disease}]")
    st.metric("Confidence", f"{prob:.2f}%")
    st.info(f"Risk Level: {risk}")

    if risk == "High":
        st.error("⚠️ IMMEDIATE DOCTOR VISIT REQUIRED")
    
    # 5. NEW: Trigger the email transfer
    if doc_email:
        report_data = {
            "name": name, "age": age, "disease": disease, 
            "prob": round(prob, 2), "risk": risk, "hr": hr, 
            "spo2": spo2, "temp": temp, "bps": bps, "bpd": bpd, "gluc": gluc
        }
        with st.spinner("Transferring report to doctor..."):
            success = send_to_doctor(doc_email, report_data)
            if success:
                st.success("Report successfully sent to doctor ✅")
