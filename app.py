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

# ---------------- CHECK FILES ----------------
required_files = ["model.pkl", "scaler.pkl", "label_encoder.pkl", "features.pkl"]

for f in required_files:
    if not os.path.exists(f):
        st.error(f"❌ Missing file: {f}")
        st.stop()

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
features = joblib.load("features.pkl")

# ---------------- LOAD MED DATA ----------------
@st.cache_data
def load_meds():
    try:
        df = pd.read_excel("Medicine_description.xlsx")
        df.columns = [c.strip() for c in df.columns]

        if 'res' in df.columns:
            df = df.rename(columns={'res': 'Reason'})

        df['Reason'] = df['Reason'].astype(str)
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

# ---------------- EMAIL ----------------
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
st.caption("ML + Explainability + Clinical Intelligence")
st.caption("Msc - Data Sciencec Project | Onkar Suresh Wagh")

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

# ---------------- RUN ----------------
if st.button("🚀 Run Diagnosis"):

    symptom_text = symptoms.lower()

    # Encode input
    symptom_vector = encode_symptoms(symptoms, features)
    vitals = [age, hr, bp, spo2, temp, gluc]

    input_data = symptom_vector + vitals
    input_df = pd.DataFrame([input_data], columns=features)

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)
    prob = model.predict_proba(scaled)

    disease = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(prob) * 100

    # ---------------- CLINICAL OVERRIDE ----------------
    if "fever" in symptom_text or temp >= 38:
        disease = "Fever"
        confidence = max(confidence, 85)

    elif "cough" in symptom_text or spo2 < 92:
        disease = "Respiratory"
        confidence = max(confidence, 80)

    elif "rash" in symptom_text:
        disease = "Allergy"

    elif gluc > 200:
        disease = "Diabetes"

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

    # ---------------- RISK & SEVERITY ----------------
    risk = 0
    if temp > 39: risk += 2
    if spo2 < 90: risk += 3
    if gluc > 200: risk += 2

    severity = "Mild"
    if risk >= 4:
        severity = "Severe"
    elif risk >= 2:
        severity = "Moderate"

    st.metric("⚠️ Risk Score", risk)
    st.metric("🔥 Severity", severity)

    # ---------------- EMAIL ----------------
    if email:
        send_email(email, name, disease, status)

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🔍 Explainability",
        "💊 Treatment"
    ])

    # -------- DASHBOARD --------
    with tab1:
        fig = px.bar(
            pd.DataFrame({
                "Disease": label_encoder.classes_,
                "Probability": prob[0]*100
            }),
            x="Probability",
            y="Disease",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- EXPLAINABILITY --------
    with tab2:
        st.subheader("Model Feature Importance")

        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(imp_df)

            fig2 = px.bar(imp_df.head(10),
                          x="Importance",
                          y="Feature",
                          orientation='h')
            st.plotly_chart(fig2)

        st.subheader("Clinical Reasoning")

        reasons = []

        if temp >= 38:
            reasons.append("High temperature indicates fever/infection")

        if spo2 < 92:
            reasons.append("Low oxygen suggests respiratory issue")

        if gluc > 200:
            reasons.append("High glucose indicates diabetes risk")

        if "cough" in symptom_text:
            reasons.append("Cough supports respiratory diagnosis")

        if "rash" in symptom_text:
            reasons.append("Rash indicates allergy")

        for r in reasons:
            st.write("✔", r)

    # -------- TREATMENT --------
    with tab3:
        st.subheader("💊 Recommended Medicines")

        search_terms = [disease.lower()]

        if "fever" in symptom_text:
            search_terms.append("fever")

        if "cough" in symptom_text:
            search_terms.append("cold")

        query = "|".join(search_terms)

        meds = med_db[
            med_db['Reason'].str.lower().str.contains(query, na=False)
        ]

        if meds.empty:
            meds = med_db.head(5)

        for _, row in meds.head(10).iterrows():
            st.markdown(f"""
            <div class='med-card'>
                <b>{row['Drug_Name']}</b><br>
                <i>{row['Reason']}</i><br>
                <small>{row['Description']}</small>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- REPORT ----------------
    report = generate_report(name, disease, round(confidence,2), status)

    st.download_button(
        "📄 Download Report",
        report,
        file_name=f"{name}_report.html",
        mime="text/html"
    )
