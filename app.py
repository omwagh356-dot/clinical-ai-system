import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. DATA & MODEL LOADING ---
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
    
    # Load Medicine data with robust settings
    file_path = 'Medicine_description.xlsx - Sheet1.csv'
    try:
        med_db = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')
        med_db.columns = med_db.columns.str.strip()
        # Find 'Reason' column even if case differs
        actual_cols = {col.lower(): col for col in med_db.columns}
        if 'reason' in actual_cols:
            med_db.rename(columns={actual_cols['reason']: 'Reason'}, inplace=True)
        med_db['Reason'] = med_db['Reason'].astype(str).str.strip().str.title()
    except:
        med_db = pd.DataFrame(columns=['Drug_Name', 'Reason', 'Description'])
        
    return disease_df, med_db

# Load all assets into global memory
model, scaler, label_encoder = load_ml_assets()
disease_db, med_db = load_clinical_data()

# --- 2. CORE LOGIC ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Ensure drug_module.py and explain.py are in your GitHub.")

def get_triage_status(disease, prob, spo2, bps):
    if spo2 < 88 or bps > 190 or prob > 98:
        return "🔴 CRITICAL", "Immediate Physician Intervention Required", "#ff4b4b"
    elif prob > 75:
        return "🟡 URGENT", "Priority Nursing Assessment", "#ffa500"
    else:
        return "🟢 STABLE", "Routine Monitoring", "#28a745"

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Advanced Clinical AI", layout="wide")
st.title("🛡️ Data-Driven Clinical Decision Support")

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
    st.subheader("🧪 Contextual Data")
    curr_diseases = st.text_area("Known Conditions / Symptoms")
    curr_drugs = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")

# --- 4. EXECUTION ---
st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    # Predict
    raw_vitals = [age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] 
    scaled = scaler.transform(pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_))
    pred = model.predict(scaled, verbose=0)
    
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100
    prob_df = pd.DataFrame({"Condition": label_encoder.classes_, "Prob": pred[0]*100}).sort_values("Prob")

    # Triage
    status, action, color = get_triage_status(disease, prob, spo2, bps)
    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Medicine Knowledge", "📄 Report"])

    with tab1:
        cg1, cg2 = st.columns([1.2, 1])
        with cg1:
            st.plotly_chart(px.bar(prob_df, x="Prob", y="Condition", orientation='h', color="Prob", color_continuous_scale="Reds", template="plotly_dark"), use_container_width=True)
        with cg2:
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose'], fill='toself'))
            fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader(f"📚 Database Research: {disease}")
        
        # --- SAFE SYMPTOM SEARCH (The Fix for your Error) ---
        # Search for the disease in your CSV
        search_term = disease.strip().title()
        matching_rows = disease_db[disease_db['Disease'].str.contains(search_term, case=False, na=False)]
        
        if not matching_rows.empty:
            typical_symptoms = matching_rows.iloc[0, 1:].dropna().unique().tolist()
            st.write("**Typical Symptoms for this Diagnosis:**")
            st.write(", ".join([s.replace('_', ' ').title() for s in typical_symptoms]))
        else:
            st.info(f"ℹ️ Predicted disease '{disease}' is not in the symptom CSV database. Showing closest matches...")

        st.divider()

        # 2. Pull drugs from Medicine_description CSV
        relevant_meds = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(10)
        if not relevant_meds.empty:
            st.write("**Pharmacological Knowledge Base:**")
            for _, row in relevant_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Action:** {row['Description']}")
        else:
            st.warning("No drugs found in local database for this predicted condition.")

    with tab3:
        st.subheader("📄 Clinical Report & Physician Communication")
        
        # 1. Prepare the Report Data
        report_payload = {
            "Name": name if name else "Unknown Patient",
            "Age": age,
            "Gender": gender,
            "Disease": disease,
            "Prob": round(prob, 2),
            "Risk": status,
            "Urgency": action,
            "Symptoms": curr_diseases if curr_diseases else "None reported",
            "vitals": raw_vitals
        }

        # 2. Generate the HTML Report (PDF alternative)
        html_doc = create_pdf_report(report_payload, info, reasons, warnings)
        
        col_dl, col_em = st.columns(2)
        
        with col_dl:
            st.download_button(
                label="📥 Download Clinical Report (HTML)",
                data=html_doc,
                file_name=f"Report_{report_payload['Name']}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col_em:
            if doc_email:
                if st.button("📧 Send Report to Doctor", use_container_width=True):
                    with st.spinner("Establishing Secure SMTP Connection..."):
                        if send_to_doctor(doc_email, report_payload, reasons, warnings):
                            st.success(f"Transmission Successful to {doc_email} ✅")
                        else:
                            st.error("Transmission Failed. Check your 'Secrets' for EMAIL_USER and EMAIL_PASS.")
            else:
                st.warning("Enter a Doctor's Email in the Identity section to enable electronic transmission.")

        # 3. Quick Summary for UI
        st.divider()
        st.markdown("### 📋 Executive Summary")
        st.write(f"**Patient:** {report_payload['Name']} | **Triage Status:** {status}")
        st.write(f"**Primary Diagnosis:** {disease} ({prob:.2f}%)")
        st.write(f"**Action Required:** {action}")
