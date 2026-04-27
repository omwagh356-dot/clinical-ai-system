import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import smtplib
from email.message import EmailMessage
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Clinical AI System", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.status-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}
.med-card {
    background: rgba(0,0,0,0.4);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL FILES ----------------
required_files = ["model.pkl", "scaler.pkl", "label_encoder.pkl", "features.pkl"]

for f in required_files:
    if not os.path.exists(f):
        st.error(f"❌ Missing file: {f}")
        st.stop()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
features = joblib.load("features.pkl")

# ---------------- LOAD MEDICINE DATA ----------------
@st.cache_data
def load_meds():
    try:
        df = pd.read_excel("Medicine_description.xlsx")
        df.columns = [c.strip() for c in df.columns]

        if 'res' in df.columns:
            df = df.rename(columns={'res': 'Reason'})

        return df
    except:
        return pd.DataFrame(columns=["Drug_Name","Reason","Description"])

med_db = load_meds()

# ---------------- SYMPTOM ENCODING ----------------
def encode_symptoms(text, feature_list):
    text = text.lower()
    vector = []

    for f in feature_list:
        if f in ['age','hr','bp','spo2','temp','glucose']:
            continue

        words = f.replace("_"," ").split()

        if any(word in text for word in words):
            vector.append(1)
        else:
            vector.append(0)

    return vector

# ---------------- EMAIL FUNCTION ----------------
def send_email(receiver, name, disease, status):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"🚨 Clinical Alert - {status}"
        msg['From'] = st.secrets["EMAIL_USER"]
        msg['To'] = receiver

        msg.set_content(f"""
Patient: {name}
Diagnosis: {disease}
Status: {status}
""")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)

        return True
    except:
        return False

# ---------------- REPORT ----------------
def generate_report(name, disease, prob, status):
    return f"""
    <html>
    <body>
    <h2>Clinical Report</h2>
    <p><b>Patient:</b> {name}</p>
    <p><b>Disease:</b> {disease}</p>
    <p><b>Confidence:</b> {prob}%</p>
    <p><b>Status:</b> {status}</p>
    </body>
    </html>
    """

# ---------------- UI ----------------
st.title("🛡️ AI Clinical Decision Support System")
st.caption("ML + Clinical Logic + Dashboard + Alerts")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120, 30)
    hr = st.number_input("Heart Rate", value=72.0)
    bp = st.number_input("Blood Pressure", value=120.0)

with col2:
    spo2 = st.number_input("SpO2", value=98.0)
    temp = st.number_input("Temperature", value=37.0)
    gluc = st.number_input("Glucose", value=90.0)
    email = st.text_input("Doctor Email")

symptoms = st.text_area("Symptoms (e.g. fever, cough, rash)")

# ---------------- RUN MODEL ----------------
if st.button("🚀 Run Diagnosis"):

    # --- Encode symptoms ---
    symptom_vector = encode_symptoms(symptoms, features)

    # --- Add vitals ---
    vitals = [age, hr, bp, spo2, temp, gluc]

    # --- Final input ---
    input_data = symptom_vector + vitals
    input_df = pd.DataFrame([input_data], columns=features)

    # --- Scale ---
    scaled = scaler.transform(input_df)

    # --- Predict ---
    prediction = model.predict(scaled)
    prob = model.predict_proba(scaled)

    disease = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(prob) * 100

    # ---------------- SMART CLINICAL LOGIC ----------------
    symptom_text = symptoms.lower()

    if confidence < 60:
        if temp >= 38:
            disease = "Fever"
        elif spo2 < 92:
            disease = "Respiratory"
        elif gluc > 200:
            disease = "Diabetes"
        elif "rash" in symptom_text:
            disease = "Allergy"

    # ---------------- STATUS ----------------
    status = "🟢 STABLE"
    if temp > 39 or spo2 < 90:
        status = "🔴 CRITICAL"

    color = "#28a745" if "STABLE" in status else "#ff4b4b"

    # ---------------- RESULT ----------------
    st.markdown(f"""
    <div class='status-box' style='background:{color};'>
        {status} : {disease} ({round(confidence,2)}%)
    </div>
    """, unsafe_allow_html=True)

    if confidence < 50:
        st.warning("⚠️ Low confidence → clinical rules applied")

    # ---------------- EMAIL ----------------
    if email:
        if send_email(email, name, disease, status):
            st.success("📧 Alert sent to doctor")
        else:
            st.error("Email failed. Check secrets.")

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 Prediction Probabilities")

    fig = px.bar(
        pd.DataFrame({
            "Disease": label_encoder.classes_,
            "Probability": prob[0]*100
        }),
        x="Probability",
        y="Disease",
        orientation='h',
        color="Probability"
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- MEDICINES ----------------
    st.subheader("💊 Recommended Medicines")

    if 'Reason' in med_db.columns:
        meds = med_db[
            med_db['Reason'].str.lower().str.contains(disease.lower(), na=False)
        ]
    else:
        meds = pd.DataFrame()

    if not meds.empty:
        for _, row in meds.head(10).iterrows():
            st.markdown(f"""
            <div class='med-card'>
                <b>{row['Drug_Name']}</b><br>
                <i>{row['Reason']}</i><br>
                <small>{row['Description']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No medicines found")

    # ---------------- REPORT DOWNLOAD ----------------
    report = generate_report(name, disease, round(confidence,2), status)

    st.download_button(
        "📄 Download Report",
        report,
        file_name=f"{name}_report.html",
        mime="text/html"
    )
