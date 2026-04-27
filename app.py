import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import smtplib
from email.message import EmailMessage

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Clinical System", layout="wide")

# ---------------- CUSTOM UI ----------------
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
    margin-bottom: 15px;
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
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
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
        st.warning("Medicine dataset not found")
        return pd.DataFrame(columns=["Drug_Name", "Reason", "Description"])

med_db = load_meds()

# ---------------- EMAIL FUNCTION ----------------
def send_email(receiver, name, disease, status):
    try:
        msg = EmailMessage()
        msg['Subject'] = f"🚨 Clinical Alert - {status}"
        msg['From'] = st.secrets["EMAIL_USER"]
        msg['To'] = receiver

        msg.set_content(f"""
Patient: {name}
Condition: {disease}
Status: {status}
""")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)

        return True
    except:
        return False

# ---------------- REPORT FUNCTION ----------------
def generate_report(name, disease, prob, status):
    return f"""
    <html>
    <body>
    <h2>Clinical Diagnostic Report</h2>
    <p><b>Patient:</b> {name}</p>
    <p><b>Disease:</b> {disease}</p>
    <p><b>Confidence:</b> {prob}%</p>
    <p><b>Status:</b> {status}</p>
    </body>
    </html>
    """

# ---------------- HEADER ----------------
st.title("🛡️ AI Clinical Decision Support System")
st.caption("Real-time Diagnosis • Smart Treatment • Explainability")

# ---------------- INPUT ----------------
st.markdown("<div class='section-title'>👤 Patient Details</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RUN MODEL ----------------
if st.button("🚀 Run Diagnosis"):

    # Create input dictionary dynamically
    input_dict = {
        'age': age,
        'hr': hr,
        'bp': bp,
        'spo2': spo2,
        'temp': temp,
        'glucose': gluc
    }

# Match model features exactly
expected_features = scaler.feature_names_in_

# Fill missing features with default values
for feature in expected_features:
    if feature not in input_dict:
        input_dict[feature] = 0

# Create DataFrame in correct order
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

    # ---------------- EMAIL ALERT ----------------
    if email:
        if send_email(email, name, disease, status):
            st.success("📧 Alert sent to doctor")
        else:
            st.error("Email failed. Check secrets.")

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Explainability", "💊 Treatment"])

    # -------- DASHBOARD --------
    with tab1:
        st.markdown("<div class='section-title'>📊 Clinical Analytics</div>", unsafe_allow_html=True)

        fig = px.bar(
            pd.DataFrame({
                "Condition": label_encoder.classes_,
                "Probability": prob[0] * 100
            }),
            x="Probability",
            y="Condition",
            orientation='h',
            color="Probability"
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig, use_container_width=True)

    # -------- EXPLAINABILITY --------
    with tab2:
        st.subheader("🔍 Feature Importance")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            imp_df = pd.DataFrame({
                "Feature": input_data.columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(imp_df)

            fig2 = px.bar(imp_df, x="Importance", y="Feature", orientation='h')
            st.plotly_chart(fig2)
        else:
            st.info("Model does not support feature importance")

    # -------- MEDICINE --------
    with tab3:
        st.markdown("<div class='section-title'>💊 Recommended Medicines</div>", unsafe_allow_html=True)

        if 'Reason' in med_db.columns:
            meds_found = med_db[
                med_db['Reason'].str.contains(disease, case=False, na=False)
            ]
        else:
            meds_found = pd.DataFrame()

        if not meds_found.empty:
            for _, row in meds_found.head(10).iterrows():
                st.markdown(f"""
                <div class='med-card'>
                    <b>{row['Drug_Name']}</b><br>
                    <i>{row['Reason']}</i><br>
                    <small>{row['Description']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No medicines found")

    # -------- REPORT DOWNLOAD --------
    report_html = generate_report(name, disease, round(confidence, 2), status)

    st.download_button(
        "📄 Download Report",
        report_html,
        file_name=f"{name}_report.html",
        mime="text/html"
    )
