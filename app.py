import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE FUNCTIONS ---

def create_clinical_report(report, reasons, warnings):
    """Generates a professional HTML report."""
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report['Name']} | <b>Age:</b> {report['Age']}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3 style='margin:0;'>Primary Diagnostic: {report['Disease']}</h3>
            <p style='color:#d93025; font-weight:bold;'>Triage Status: {report['Risk']}</p>
        </div>
        <h4>📊 Physiological Analysis</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
        <p><b>Recommended Action:</b> {report['Urgency']}</p>
    </div>
    """

def send_to_physician(receiver, report, reasons):
    """Sends a secure clinical alert via email."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Clinical Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    body = f"Clinical Handover for {report['Name']}.\nPredicted: {report['Disease']}\nAnalysis: {reasons}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 2. DATA LOADING (Handles your 'res' column automatically) ---

@st.cache_data
def load_clinical_data():
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    try:
        # Renaming the 'res' column from your Medicine file to 'Reason' internally
        df = pd.read_csv('Medicine_description.xlsx - Sheet1.csv', sep=None, engine='python', encoding='latin1')
        df.columns = [col.strip().replace('ï»¿', '') for col in df.columns]
        if 'res' in df.columns: df = df.rename(columns={'res': 'Reason'})
        else: df = df.rename(columns={df.columns[1]: 'Reason'})
        df['Reason'] = df['Reason'].fillna('Unknown').astype(str).str.strip().str.title()
        return disease_df, df
    except:
        return disease_df, pd.DataFrame(columns=['Drug_Name', 'Reason', 'Description'])

@st.cache_resource
def load_ml_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

# Logic imports
try:
    from drug_module import check_drugs
    from explain import explain_values
except:
    st.error("Missing logic modules.")

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Advanced CDSS", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Patient Full Name")
    doc_email = st.text_input("Physician Email")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", 1, 120, 30)
    
    st.subheader("📉 Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    bpd = v2.number_input("BP Diastolic", value=80.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)

with c2:
    st.subheader("🧪 Contextual Symptoms")
    curr_syms = st.text_area("Symptoms (e.g. fever, cough, runny nose, acne)")
    curr_meds = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")

# --- 4. THE DIAGNOSTIC ENGINE ---

st.divider()
if st.button("🚀 EXECUTE FULL SCAN", type="primary", use_container_width=True):
    # Predict with Neural Network
    raw_vitals = [age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] 
    scaled = scaler.transform(pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_))
    prediction = model.predict(scaled, verbose=0)
    
    idx = np.argmax(prediction)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = prediction[0][idx] * 100

    # --- MULTIMODAL OVERRIDE (This handles Fever/Cough/Runny Nose) ---
    symptom_text = curr_syms.lower()
    
    # 1. Check for Respiratory/Fever Cluster
    if temp >= 38.5 or "fever" in symptom_text or "cough" in symptom_text or "runny nose" in symptom_text:
        if disease == "Normal":
            disease = "Common Cold / Flu / Respiratory Infection"
            prob = 98.0  # High confidence based on clinical evidence
    
    # 2. Check for Skin Cluster
    if "pimple" in symptom_text or "acne" in symptom_text:
        if disease == "Normal":
            disease = "Acne Vulgaris"
            prob = 95.0

    # Triage Logic
    status = "🔴 CRITICAL" if temp >= 39.5 or spo2 < 89 or bps > 185 else "🟢 STABLE"
    color = "#ff4b4b" if status == "🔴 CRITICAL" else "#28a745"
    action = "Immediate Medical Intervention" if status == "🔴 CRITICAL" else "Routine Monitoring"
    
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_meds.split(","), curr_syms.split(","), curr_allergies.split(","))

    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Therapy pathway", "📄 Report & Email"])
    
    with tab1:
        g1, g2 = st.columns([1.3, 1])
        with g1:
            st.plotly_chart(px.bar(pd.DataFrame({"Condition": label_encoder.classes_, "Prob": prediction[0]*100}), x="Prob", y="Condition", orientation='h', template="plotly_dark"), use_container_width=True)
        with g2:
            # Radar chart specifically shows the spike for the high temp
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP', 'Temp', 'Gluc'], fill='toself'))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Pharmaceutical Database: {disease}")
        # Handle search for multi-word diseases
        search_key = "Common Cold" if "Cold" in disease else disease
        rel_meds = med_db[med_db['Reason'].str.contains(search_key.split()[0], case=False, na=False)].head(10)
        
        if not rel_meds.empty:
            for _, row in rel_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Description:** {row['Description']}")
        else:
            st.warning("No drugs found in local database for this predicted condition.")

    with tab3:
        report_data = {"Name": name, "Age": age, "Gender": gender, "Disease": disease, "Prob": round(prob, 2), "Risk": status, "Urgency": action}
        html_report = create_clinical_report(report_data, reasons, warnings)
        
        st.download_button("📥 Download Official Report", html_report, file_name=f"Clinical_{name}.html", mime="text/html", use_container_width=True)
        
        if doc_email and st.button("📧 Email Physician"):
            if send_to_physician(doc_email, report_data, reasons):
                st.success("Sent Successfully! ✅")
