import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE FUNCTIONS (Must be at top to avoid NameErrors) ---

def create_clinical_report(report, reasons, warnings):
    """Generates a professional HTML diagnostic report."""
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']} | <b>Gender:</b> {report['Gender']}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3 style='margin:0;'>AI Prediction: {report['Disease']} ({report['Prob']}%)</h3>
            <p style='color:#d93025; font-weight:bold;'>Triage Status: {report['Risk']}</p>
        </div>
        <h4>📊 Clinical Vitals Analysis</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
        <h4>💊 Safety Alerts</h4>
        <ul>{"".join([f"<li>⚠️ {w}</li>" for w in warnings]) if warnings else "<li>No immediate drug-vital risks flagged</li>"}</ul>
        <p><b>Action Plan:</b> {report['Urgency']}</p>
    </div>
    """

def send_to_physician(receiver, report, reasons):
    """Handles secure SMTP transmission to the doctor."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Clinical Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    
    body = f"""OFFICIAL CLINICAL REPORT
-------------------------------
PATIENT: {report['Name']} ({report['Age']}yr {report['Gender']})
PREDICTION: {report['Disease']} ({report['Prob']}%)
TRIAGE: {report['Risk']}

VITAL ANALYSIS:
{chr(10).join(['- ' + r for r in reasons])}
-------------------------------
Generated via Clinical AI CDSS."""
    
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except:
        return False

# --- 2. DATA & ASSET LOADING ---

@st.cache_data
def load_clinical_data():
    # Load Disease Knowledge
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    
    # Load Medicine Knowledge (Handling your 'res' column)
    try:
        med_df = pd.read_csv('Medicine_description.xlsx', sep=None, engine='python', encoding='latin1')
        med_df.columns = [col.strip().replace('ï»¿', '') for col in med_df.columns]
        
        # Mapping 'res' or position 1 to 'Reason'
        if 'res' in med_df.columns:
            med_df = med_df.rename(columns={'res': 'Reason'})
        else:
            med_df = med_df.rename(columns={med_df.columns[1]: 'Reason'})
            
        med_df['Reason'] = med_df['Reason'].fillna('Unknown').astype(str).str.strip().str.title()
    except:
        med_df = pd.DataFrame(columns=['Drug_Name', 'Reason', 'Description'])
    return disease_df, med_df

@st.cache_resource
def load_ml_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

# Initialize
model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

try:
    from drug_module import check_drugs
    from explain import explain_values
except:
    st.error("Missing Logic Modules.")

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Advanced CDSS", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns([1, 1.2])

with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor's Email")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", 1, 120, 30)
    
    st.subheader("📉 Clinical Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bpd = v2.number_input("BP Diastolic", value=80.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)

with c2:
    st.subheader("🧪 Contextual Data")
    curr_symptoms = st.text_area("Reported Symptoms (comma separated)")
    curr_meds = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")

# --- 4. EXECUTION ENGINE ---
st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    # ML Prediction
    raw_vitals = [age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] 
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # Triage Logic
    status = "🔴 CRITICAL" if spo2 < 88 or bps > 190 else "🟢 STABLE"
    color = "#ff4b4b" if status == "🔴 CRITICAL" else "#28a745"
    action = "Immediate physician intervention required" if status == "🔴 CRITICAL" else "Routine monitoring"
    
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_meds.split(","), curr_symptoms.split(","), curr_allergies.split(","))

    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Therapy", "📄 Report & Email"])
    
    with tab1:
        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.plotly_chart(px.bar(pd.DataFrame({"Condition": label_encoder.classes_, "Prob": pred[0]*100}), x="Prob", y="Condition", orientation='h', template="plotly_dark"), use_container_width=True)
        with g2:
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose'], fill='toself'))
            fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader(f"Knowledge Base: {disease}")
        rel_meds = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(10)
        for _, row in rel_meds.iterrows():
            with st.expander(f"💊 {row['Drug_Name']}"):
                st.write(row['Description'])

    with tab3:
        report_data = {"Name": name, "Age": age, "Gender": gender, "Disease": disease, "Prob": round(prob, 2), "Risk": status, "Urgency": action}
        html_report = create_clinical_report(report_data, reasons, warnings)
        
        st.download_button("📥 Download Physician Report (HTML)", html_report, file_name=f"{name}_Report.html", mime="text/html", use_container_width=True)
        
        if doc_email and st.button("📧 Email Physician Now", use_container_width=True):
            if send_to_physician(doc_email, report_data, reasons):
                st.success(f"Report transmitted to {doc_email}! ✅")
            else:
                st.error("Email failed. Check your Secrets.")
