import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import smtplib
from email.message import EmailMessage
from explain import init_explainer, get_shap_values

# --- LOAD MODEL ---

@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    # dummy data for SHAP initialization
    sample = pd.DataFrame(np.random.rand(50, len(scaler.feature_names_in_)), columns=scaler.feature_names_in_)
    init_explainer(model, sample)

    return model, scaler, le

model, scaler, label_encoder = load_assets()

# --- LOAD MED DATA ---
@st.cache_data
def load_meds():
    return pd.read_csv("Medicine_description.csv")

med_db = load_meds()

# --- EMAIL FUNCTION ---
def send_email(email, disease, risk):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"🚨 {risk} Alert"
        msg['From'] = st.secrets["EMAIL_USER"]
        msg['To'] = email
        msg.set_content(f"Diagnosis: {disease}")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except:
        return False

# --- UI ---
st.set_page_config(layout="wide")
st.title("🛡️ AI Clinical Decision Support System")

c1, c2 = st.columns([1,1.2])

with c1:
    name = st.text_input("Patient Name")
    email = st.text_input("Doctor Email")
    age = st.number_input("Age", 1, 120)

    hr = st.number_input("Heart Rate", 40, 180)
    bp = st.number_input("BP", 80, 200)
    spo2 = st.number_input("SpO2", 70, 100)
    temp = st.number_input("Temperature", 35.0, 42.0)
    gluc = st.number_input("Glucose", 50, 400)

with c2:
    symptoms = st.text_area("Symptoms")
    meds = st.text_area("Current Medications")

# --- RUN MODEL ---
if st.button("🚀 Run Diagnosis"):

    input_df = pd.DataFrame([{
        "Age": age,
        "HeartRate": hr,
        "BP_Systolic": bp,
        "SpO2": spo2,
        "Temperature": temp,
        "Glucose": gluc
    }])

    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    scaled = scaler.transform(input_df)

    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)

    disease = label_encoder.inverse_transform(pred)[0]
    confidence = np.max(prob) * 100

    # Risk Score
    risk = 0
    if hr > 120: risk += 2
    if spo2 < 90: risk += 3
    if temp > 39: risk += 3
    if gluc > 250: risk += 2

    status = "🔴 CRITICAL" if risk >= 5 else "🟢 STABLE"

    st.success(f"{status} - {disease} ({confidence:.2f}%)")

    tabs = st.tabs(["📊 Dashboard", "🔍 Explainability", "💊 Treatment"])

    # ---------------- DASHBOARD ----------------
    with tabs[0]:
        st.subheader("Prediction Distribution")

        df_plot = pd.DataFrame({
            "Disease": label_encoder.classes_,
            "Probability": prob[0]*100
        })

        st.plotly_chart(px.bar(df_plot, x="Probability", y="Disease"))

        st.subheader("Vitals Radar")
        radar = [hr/150, bp/180, (100-spo2)/20, abs(temp-37)/5, gluc/300]

        fig = go.Figure(go.Scatterpolar(
            r=radar,
            theta=["HR","BP","SpO2","Temp","Glucose"],
            fill='toself'
        ))
        st.plotly_chart(fig)

        # Export for Power BI
        export_df = input_df.copy()
        export_df["Prediction"] = disease
        export_df.to_csv("dashboard_data.csv", index=False)

        st.download_button("⬇️ Export for Power BI", export_df.to_csv(index=False), "dashboard_data.csv")

    # ---------------- SHAP ----------------
    with tabs[1]:
        st.subheader("🔍 Model Explainability (SHAP)")

        shap_values = get_shap_values(input_df)

        st.write("Feature Impact:")
        shap_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Impact": shap_values.values[0]
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.dataframe(shap_df)

        fig = px.bar(shap_df, x="Impact", y="Feature", orientation='h')
        st.plotly_chart(fig)

    # ---------------- TREATMENT ----------------
    with tabs[2]:
        st.subheader("💊 Recommended Medicines")

        meds_found = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)]

        if not meds_found.empty:
            for _, row in meds_found.head(5).iterrows():
                st.write(f"**{row['Drug_Name']}** - {row['Description']}")
        else:
            st.warning("No medicines found")

    # --- EMAIL ---
    if status == "🔴 CRITICAL" and email:
        if send_email(email, disease, status):
            st.success("Alert sent to doctor")
