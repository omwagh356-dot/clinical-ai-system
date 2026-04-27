import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import smtplib
from email.message import EmailMessage

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Clinical System", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
}
.section-title {
    font-size: 22px;
    font-weight: bold;
}
.status-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
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

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------- LOAD MED DATA ----------------
@st.cache_data
def load_meds():
    try:
        df = pd.read_excel("Medicine_description.xlsx")
        df.columns = [col.strip() for col in df.columns]

        if 'res' in df.columns:
            df = df.rename(columns={'res': 'Reason'})

        return df
    except:
        return pd.DataFrame(columns=["Drug_Name", "Reason", "Description"])

med_db = load_meds()

# ---------------- EMAIL ----------------
def send_email(receiver, name, disease, status):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"🚨 Clinical Alert - {status}"
        msg['From'] = st.secrets["EMAIL_USER"]
        msg['To'] = receiver

        msg.set_content(f"""
Patient: {name}
Disease: {disease}
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

# ---------------- HEADER ----------------
st.title("🛡️ AI Clinical Decision Support System")
st.caption("Diagnosis • Dashboard • Explainability • Alerts")

# ---------------- INPUT ----------------
st.markdown("<div class='section-title'>👤 Patient Details</div>", unsafe_allow_html=True)

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

symptoms = st.text_area("Symptoms")

# ---------------- RUN ----------------
if st.button("🚀 Run Diagnosis"):

    # ---- FIXED INPUT MATCHING ----
    input_dict = {
        'age': age,
        'hr': hr,
        'bp': bp,
        'spo2': spo2,
        'temp': temp,
        'glucose': gluc
    }

    expected_features = scaler.feature_names_in_

    for f in expected_features:
        if f not in input_dict:
            input_dict[f] = 0

    input_data = pd.DataFrame([input_dict])[expected_features]

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)
    prob = model.predict_proba(scaled)

    disease = label_encoder.inverse_transform(prediction)[0]
    confidence = np.max(prob) * 100

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

    # ---------------- EMAIL ----------------
    if email:
        if send_email(email, name, disease, status):
            st.success("📧 Email sent")
        else:
            st.error("Email failed (check secrets)")

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 Prediction Breakdown")

    fig = px.bar(
        pd.DataFrame({
            "Condition": label_encoder.classes_,
            "Probability": prob[0]*100
        }),
        x="Probability",
        y="Condition",
        orientation='h'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- EXPLAINABILITY ----------------
    st.subheader("🔍 Feature Importance")

    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": expected_features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(imp_df)
    else:
        st.info("Model has no feature importance")

    # ---------------- MEDICINES ----------------
    st.subheader("💊 Medicines")

    if 'Reason' in med_db.columns:
        meds = med_db[
            med_db['Reason'].str.contains(disease, case=False, na=False)
        ]
    else:
        meds = pd.DataFrame()

    if not meds.empty:
        for _, row in meds.head(10).iterrows():
            st.markdown(f"""
            <div class='med-card'>
            <b>{row['Drug_Name']}</b><br>
            {row['Reason']}<br>
            <small>{row['Description']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No medicines found")

    # ---------------- REPORT ----------------
    report = generate_report(name, disease, round(confidence,2), status)

    st.download_button(
        "📄 Download Report",
        report,
        file_name="report.html",
        mime="text/html"
    )
