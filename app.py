import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE UTILITY FUNCTIONS (Defined first to avoid NameErrors) ---

def create_pdf_report(report, reasons, warnings):
    """Generates a professional HTML report structure."""
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report.get('Name', 'N/A')} | <b>Age:</b> {report.get('Age', 'N/A')} | <b>Gender:</b> {report.get('Gender', 'N/A')}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3 style='margin:0;'>AI Prediction: {report.get('Disease', 'Unknown')} ({report.get('Prob', 0)}%)</h3>
            <p style='color:#d93025; font-weight:bold;'>Status: {report.get('Risk', 'N/A')}</p>
        </div>
        <h4>📊 Clinical Vitals Analysis</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
        <h4>💊 Safety Alerts</h4>
        <ul>{"".join([f"<li>⚠️ {w}</li>" for w in warnings]) if warnings else "<li>No immediate drug-vital risks flagged</li>"}</ul>
        <p><b>Action Plan:</b> {report.get('Urgency', 'Consult MD.')}</p>
    </div>
    """

def send_to_doctor(receiver, report, reasons, warnings):
    """Handles secure SMTP transmission."""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Clinical Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    body = f"PATIENT: {report['Name']}\nDIAGNOSIS: {report['Disease']} ({report['Prob']}%)\nVITALS: {reasons}"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

def get_triage_status(disease, prob, spo2, bps):
    if spo2 < 88 or bps > 190 or (disease != "Normal" and prob > 92):
        return "🔴 CRITICAL", "Immediate Physician Intervention Required", "#ff4b4b"
    return ("🟡 URGENT", "Priority Assessment", "#ffa500") if prob > 75 else ("🟢 STABLE", "Routine Monitoring", "#28a745")

# --- 2. DATA & ASSET LOADING ---

@st.cache_resource
def load_ml_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

@st.cache_data
def load_clinical_data():
    # Load Disease symptoms
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    
    # Load Medicine data with aggressive header cleaning to fix KeyError: 'Reason'
    try:
        med_df = pd.read_csv('Medicine_description.xlsx', encoding='latin1', on_bad_lines='skip', engine='python')
        # Strip invisible Byte Order Marks (ï»¿) and whitespace
        med_df.columns = [col.strip().replace('ï»¿', '') for col in med_df.columns]
        
        # Standardize 'Reason' column name
        col_map = {col.lower(): col for col in med_df.columns}
        if 'reason' in col_map:
            med_df = med_df.rename(columns={col_map['reason']: 'Reason'})
        else:
            # Fallback: assume the 2nd column is Reason if exact name not found
            med_df = med_df.rename(columns={med_df.columns[1]: 'Reason'})
            
        med_df['Reason'] = med_df['Reason'].fillna('Unknown').astype(str).str.strip().str.title()
    except:
        med_df = pd.DataFrame(columns=['Drug_Name', 'Reason', 'Description'])
        
    return disease_df, med_df

# Global Initialization
model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Ensure drug_module.py and explain.py are in your directory.")

# --- 3. UI LAYOUT ---

st.set_page_config(page_title="Advanced Clinical AI", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns(2)
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
    st.subheader("🧪 History & Symptoms")
    curr_diseases = st.text_area("Symptoms (e.g. skin_rash, chills)")
    curr_drugs = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")
    
    if st.button("PRE-CHECK DRUG SAFETY", width="stretch"):
        w_p, r_p = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
        for w in w_p: st.error(w)
        for r in r_p: st.success(r)

# --- 4. DIAGNOSTIC EXECUTION ---

st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", width="stretch"):
    # ML Prediction
    raw_vitals = [age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] 
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100
    prob_df = pd.DataFrame({"Condition": label_encoder.classes_, "Prob": pred[0]*100}).sort_values("Prob")

    # Analytics & Logic
    status, action, color = get_triage_status(disease, prob, spo2, bps)
    reasons = explain_values(hr, (bps+bpd)/2, spo2, temp)
    warnings, _ = check_drugs(curr_drugs.split(","), curr_diseases.split(","), curr_allergies.split(","))
    
    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Medicine Knowledge", "📄 Report & Email"])
    
    with tab1:
        cv1, cv2 = st.columns([1.2, 1])
        with cv1:
            st.plotly_chart(px.bar(prob_df, x="Prob", y="Condition", orientation='h', color="Prob", color_continuous_scale="Reds", template="plotly_dark"), width="stretch")
        with cv2:
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose'], fill='toself'))
            fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, width="stretch")

    with tab2:
        st.subheader(f"📚 Knowledge Base: {disease}")
        # Safe Symptom Search
        search_term = disease.strip().title()
        match = disease_db[disease_db['Disease'].str.contains(search_term, case=False, na=False)]
        if not match.empty:
            typical_symptoms = match.iloc[0, 1:].dropna().unique().tolist()
            st.write("**Typical Symptoms:**", ", ".join([s.replace('_', ' ').title() for s in typical_symptoms]))
        
        st.divider()
        
        # Safe Medicine Search
        if 'Reason' in med_db.columns:
            rel_meds = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(10)
            if not rel_meds.empty:
                for _, row in rel_meds.iterrows():
                    with st.expander(f"💊 {row.get('Drug_Name', 'Drug')}"):
                        st.write(f"**Description:** {row.get('Description', 'N/A')}")
            else:
                st.warning("No drugs found for this condition in the local database.")

    with tab3:
        report_data = {
            "Name": name if name else "Patient", "Age": age, "Gender": gender, 
            "Disease": disease, "Prob": round(prob, 2), "Risk": status, 
            "Urgency": action, "vitals": raw_vitals
        }
        
        html_doc = create_pdf_report(report_data, reasons, warnings)
        
        cdl, cem = st.columns(2)
        with cdl:
            st.download_button("📥 Download Report (HTML)", html_doc, file_name=f"{name}_Report.html", mime="text/html", width="stretch")
        with cem:
            if doc_email and st.button("📧 Email Report", width="stretch"):
                if send_to_doctor(doc_email, report_data, reasons, warnings):
                    st.success("Sent to Physician Successfully! ✅")
