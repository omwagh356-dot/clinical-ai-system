import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. CORE UTILITY FUNCTIONS ---

def create_pdf_report(report, reasons, warnings):
    return f"""
    <div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
        <h1 style='color:#1a73e8; text-align:center;'>Clinical AI Diagnostic Report</h1>
        <hr>
        <p><b>Patient:</b> {report.get('Name', 'N/A')} | <b>Age:</b> {report.get('Age', 'N/A')}</p>
        <div style='background:#f0f2f6; padding:15px; border-radius:10px;'>
            <h3>AI Prediction: {report.get('Disease', 'Unknown')}</h3>
        </div>
        <h4>📊 Analysis</h4>
        <ul>{"".join([f"<li>{r}</li>" for r in reasons])}</ul>
    </div>
    """

def send_to_doctor(receiver, report, reasons, warnings):
    msg = EmailMessage()
    msg['Subject'] = f"🚨 {report['Risk']} Alert - {report['Name']}"
    msg['From'] = st.secrets.get("EMAIL_USER", "")
    msg['To'] = receiver
    msg.set_content(f"Patient {report['Name']} diagnosed with {report['Disease']}.")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except: return False

# --- 2. DATA LOADING (UPDATED FOR 'res' COLUMN) ---

@st.cache_data
def load_clinical_data():
    # 1. Load Disease symptoms
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    
    # 2. Load Medicine data with Auto-Detection
    file_path = 'Medicine_description.xlsx'
    try:
        # 'sep=None' automatically detects if the file uses , or ; or \t
        med_df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
        
        # Clean the column headers (removes invisible spaces or BOM characters)
        med_df.columns = [col.strip().replace('ï»¿', '') for col in med_df.columns]
        
        # SEARCH FOR THE COLUMN: We check for 'res' or 'Reason' or the 2nd position
        if 'res' in med_df.columns:
            med_df = med_df.rename(columns={'res': 'Reason'})
        elif 'Reason' in med_df.columns:
            pass # Already named correctly
        elif len(med_df.columns) > 1:
            # Fallback: Rename the second column found to 'Reason'
            med_df = med_df.rename(columns={med_df.columns[1]: 'Reason'})
        else:
            st.error("The medicine file appears to have only one column. Check your CSV formatting.")

        # Final cleanup to ensure strings work in search
        med_df['Reason'] = med_df['Reason'].fillna('Unknown').astype(str).str.strip().str.title()
        
    except Exception as e:
        st.error(f"Critical Error loading medicine file: {e}")
        med_df = pd.DataFrame(columns=['Drug_Name', 'Reason', 'Description'])
        
    return disease_df, med_df
@st.cache_resource
def load_ml_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("Logic modules missing.")

# --- 3. UI & LOGIC ---

st.set_page_config(page_title="Clinical AI", layout="wide")
st.title("🛡️ Clinical AI Dashboard")

c1, c2 = st.columns(2)
with c1:
    name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor Email")
    age = st.number_input("Age", 1, 120, 30)
    hr = st.number_input("Heart Rate", value=72.0)
    bps = st.number_input("BP Systolic", value=120.0)
    spo2 = st.number_input("SpO2 %", value=98.0)
    bpd = st.number_input("BP Diastolic", value=80.0)
    temp = st.number_input("Temp", value=37.0)
    gluc = st.number_input("Glucose", value=95.0)

with c2:
    curr_diseases = st.text_area("Symptoms")
    curr_drugs = st.text_area("Meds")
    curr_allergies = st.text_area("Allergies")

if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", width="stretch"):
    # Prediction
    raw_vitals = [age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] 
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100
    
    if spo2 < 88 or bps > 190: status, color = "🔴 CRITICAL", "#ff4b4b"
    else: status, color = "🟢 STABLE", "#28a745"

    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Medicine Base", "📄 Report"])
    
    with tab1:
        st.plotly_chart(px.bar(pd.DataFrame({"Condition": label_encoder.classes_, "Prob": pred[0]*100}), x="Prob", y="Condition", orientation='h', template="plotly_dark"), width="stretch")

    with tab2:
        st.subheader(f"Medicine Knowledge for {disease}")
        # This will now work because we renamed 'res' to 'Reason' in the loader
        rel_meds = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(10)
        
        if not rel_meds.empty:
            for _, row in rel_meds.iterrows():
                with st.expander(f"💊 {row.get('Drug_Name')}"):
                    st.write(f"**Indication:** {row.get('Reason')}")
                    st.write(f"**Description:** {row.get('Description')}")
        else:
            st.warning("No drugs found for this condition.")

    with tab3:
        report_data = {"Name": name, "Disease": disease, "Prob": round(prob, 2), "Risk": status, "Urgency": "Follow-up required."}
        html = create_pdf_report(report_data, ["Vitals within normal limits" if status == "🟢 STABLE" else "Abnormal vitals detected"], [])
        st.download_button("📥 Download Report", html, file_name="Clinical_Report.html", mime="text/html", width="stretch")
