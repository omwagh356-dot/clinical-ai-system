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

def get_triage_status(disease, prob, spo2, bps):
    """Innovative Triage Logic: Determines urgency based on AI + Vitals."""
    if spo2 < 88 or bps > 190 or (disease != "Normal" and prob > 90):
        return "🔴 CRITICAL", "Immediate Physician Intervention Required", "#ff4b4b"
    elif prob > 70:
        return "🟡 URGENT", "Priority Nursing Assessment", "#ffa500"
    else:
        return "🟢 STABLE", "Routine Monitoring", "#28a745"

def create_pdf_report(report, reasons, warnings):
    return f"""<div style='font-family:Arial; border:2px solid #333; padding:20px; border-radius:10px;'>
    <h2 style='color:#1a73e8;'>Clinical AI Diagnostic Report</h2>
    <hr>
    <p><b>Patient:</b> {report.get('Name')} | <b>Age:</b> {report.get('Age')}</p>
    <p><b>Diagnosis:</b> {report.get('Disease')} ({report.get('Prob')}%)</p>
    <p><b>Triage:</b> {report.get('Risk')}</p>
    </div>"""

# --- 2. DATA & ASSET LOADING ---

@st.cache_data
def load_clinical_data():
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    
    try:
        # Auto-detecting the 'res' column you created
        med_df = pd.read_csv('Medicine_description.xlsx', sep=None, engine='python', encoding='latin1')
        med_df.columns = [col.strip().replace('ï»¿', '') for col in med_df.columns]
        
        # Mapping your new 'res' column to 'Reason' internally
        if 'res' in med_df.columns:
            med_df = med_df.rename(columns={'res': 'Reason'})
        elif len(med_df.columns) > 1:
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

model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Advanced Clinical AI", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns(2)
with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    age = st.number_input("Age", 1, 120, 30)
    hr = st.number_input("Heart Rate", value=72.0)
    bps = st.number_input("BP Systolic", value=120.0)
    spo2 = st.number_input("SpO2 %", value=98.0)
    temp = st.number_input("Temp °C", value=37.0)
    gluc = st.number_input("Glucose", value=95.0)

with c2:
    st.subheader("🧪 Contextual History")
    curr_diseases = st.text_area("Symptoms (e.g. skin_rash, itching)")
    curr_drugs = st.text_area("Current Medications")
    
    # Innovative Search Tool: Quick Drug Lookup
    st.markdown("🔍 **Quick Med Search**")
    search_q = st.text_input("Search 22,000+ Drugs by Condition")
    if search_q:
        res = med_db[med_db['Reason'].str.contains(search_q, case=False, na=False)].head(3)
        for _, r in res.iterrows():
            st.caption(f"💊 {r['Drug_Name']}: {r['Description'][:100]}...")

# --- 4. ANALYTICS ENGINE ---
st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    # ML Prediction
    raw_vitals = [age, hr, bps, 80.0, spo2, temp, 190.0, gluc, 16.0] 
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100
    
    # Unique Feature: Triage Badge
    status, action, color = get_triage_status(disease, prob, spo2, bps)
    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics & Radar", "💊 Therapy Pathway", "📄 Clinical Report"])
    
    with tab1:
        g1, g2 = st.columns([1.5, 1])
        with g1:
            st.markdown("### 🎯 Differential Diagnosis (AI Confidence)")
            prob_df = pd.DataFrame({"Condition": label_encoder.classes_, "Prob": pred[0]*100}).sort_values("Prob")
            fig_bar = px.bar(prob_df, x="Prob", y="Condition", orientation='h', color="Prob", color_continuous_scale="Reds", template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with g2:
            st.markdown("### 🕸️ Physiological Fingerprint")
            # Unique Insight: Radar chart showing which vitals are most abnormal
            radar_cats = ['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose']
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=radar_cats, fill='toself', line_color=color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader(f"Drug Recommendations for {disease}")
        # Knowledge Base Search
        rel_meds = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(10)
        if not rel_meds.empty:
            for _, row in rel_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Clinical Description:** {row['Description']}")
        else:
            st.warning("No specific drugs found in local database.")

    with tab3:
        report_data = {"Name": name, "Age": age, "Disease": disease, "Prob": round(prob, 2), "Risk": status}
        html = create_pdf_report(report_data, [], [])
        st.download_button("📥 Download Physician Report", html, file_name=f"{name}_Report.html", mime="text/html", use_container_width=True)
