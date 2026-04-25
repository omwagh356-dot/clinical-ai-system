import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. ASSET LOADING (CSVs & MODELS) ---
# --- 1. DEFINE THE LOADING FUNCTION ---
@st.cache_resource
def load_ml_assets():
    model = load_model("model/model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

# --- 2. EXECUTE THE FUNCTION AT THE TOP LEVEL ---
# This makes 'scaler', 'model', and 'label_encoder' available to the whole app
model, scaler, label_encoder = load_ml_assets()

# --- 3. NOW YOUR BUTTON LOGIC WILL WORK ---
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC"):
    # This line will no longer fail because 'scaler' is defined above!
    scaled = scaler.transform(inputs_df)
@st.cache_data

def load_clinical_data():
    # 1. Load Disease symptoms (Usually UTF-8 is fine here)
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    
    # 2. Load Medicine data with Encoding Fix
    # We use 'latin1' or 'cp1252' to handle Excel-style special characters
    try:
        med_db = pd.read_csv('Medicine_description.xlsx', encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, use latin1 which is more forgiving
        med_db = pd.read_csv('Medicine_description.xlsx', encoding='latin1')

    # Clean strings to ensure matching works
    med_db['Reason'] = med_db['Reason'].str.strip().str.title()
    disease_df['Disease'] = disease_df['Disease'].str.strip().str.title()
    
    return disease_df, med_db

# --- 2. CORE LOGIC ---
try:
    from drug_module import check_drugs
    from explain import explain_values
except ImportError:
    st.error("⚠️ Ensure drug_module.py and explain.py are in the folder.")

def get_triage_status(disease, prob, spo2, bps):
    if spo2 < 88 or bps > 190 or prob > 95:
        return "🔴 CRITICAL", "Immediate Physician Intervention Required", "#ff4b4b"
    elif prob > 75:
        return "🟡 URGENT", "Priority Nursing Assessment", "#ffa500"
    else:
        return "🟢 STABLE", "Routine Monitoring", "#28a745"

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Data-Driven Clinical AI", layout="wide")
st.title("🏥 Enterprise Clinical Decision Support System")
st.caption("Powered by 22,000+ Medicine Records and 4,900+ Disease Mappings")

col_id, col_hist = st.columns(2)
with col_id:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Full Name")
    doc_email = st.text_input("Doctor's Email")
    g_col, a_col = st.columns(2)
    gender = g_col.selectbox("Gender", ["Male", "Female", "Other"])
    age = a_col.number_input("Age", 1, 120, 30)
    
    st.subheader("📉 Clinical Vitals")
    v_c1, v_c2, v_c3 = st.columns(3)
    hr = v_c1.number_input("Heart Rate", value=72.0)
    bps = v_c1.number_input("BP Systolic", value=120.0)
    spo2 = v_c2.number_input("SpO2 %", value=98.0)
    bpd = v_c2.number_input("BP Diastolic", value=80.0)
    temp = v_c3.number_input("Temp °C", value=37.0)
    gluc = v_c3.number_input("Glucose", value=95.0)

with col_hist:
    st.subheader("🧪 Patient Context")
    curr_diseases = st.text_area("Reported Symptoms (e.g., itching, chills, fatigue)")
    curr_drugs = st.text_area("Current Medications")
    curr_allergies = st.text_area("Known Allergies")

# --- 4. DIAGNOSTIC EXECUTION ---
st.divider()
if st.button("🚀 EXECUTE MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    # ML Prediction
    raw_vitals = [age, hr, bps, bpd, spo2, temp, 190.0, gluc, 16.0] # 190 and 16 as defaults
    inputs_df = pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_)
    scaled = scaler.transform(inputs_df)
    pred = model.predict(scaled, verbose=0)
    idx = np.argmax(pred)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = pred[0][idx] * 100

    # Triage
    status, action, color = get_triage_status(disease, prob, spo2, bps)
    st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1><p>{action}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Medicine Knowledge Base", "📄 Clinical Report"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Differential Diagnosis**")
            prob_df = pd.DataFrame({"Condition": label_encoder.classes_, "Prob": pred[0]*100}).sort_values("Prob")
            st.plotly_chart(px.bar(prob_df, x="Prob", y="Condition", orientation='h', color="Prob", color_continuous_scale="Reds", template="plotly_dark"), use_container_width=True)
        with c2:
            st.write("**Physiological Radar**")
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP Sys', 'Temp', 'Glucose'], fill='toself'))
            fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader(f"📚 Knowledge Base: {disease}")
        
        # 1. Pull typical symptoms from DiseaseAndSymptoms.csv
        typical_symptoms = disease_db[disease_db['Disease'].str.contains(disease, case=False, na=False)].iloc[0, 1:].dropna().unique().tolist()
        st.write("**Typical Symptoms for this Diagnosis:**")
        st.write(", ".join([s.replace('_', ' ').title() for s in typical_symptoms if str(s) != 'nan']))

        st.divider()

        # 2. Pull real drugs from Medicine_description CSV
        # We search for drugs where the 'Reason' matches the predicted disease
        relevant_meds = med_db[med_db['Reason'].str.contains(disease, case=False, na=False)].head(10)
        
        if not relevant_meds.empty:
            st.write(f"**Pharmacological Recommendations (Top {len(relevant_meds)} Matches):**")
            for _, row in relevant_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Description:** {row['Description']}")
        else:
            st.warning("No specific drugs found in local database for this predicted condition. Please consult a senior MD.")


    with tab3:
        report = {"Name": name, "Age": age, "Gender": gender, "Disease": disease, "Prob": round(prob, 2), "Risk": status, "Urgency": action, "Symptoms": curr_diseases, "vitals": raw_vitals}
        html_doc = create_pdf_report(report, info, reasons, warnings)
        st.download_button("Download Full Report (HTML)", html_doc, file_name=f"{name}_Report.html", mime="text/html", use_container_width=True)
        
        if doc_email:
            with st.spinner("Transmitting Clinical Data..."):
                if send_to_doctor(doc_email, report, reasons, warnings):
                    st.success(f"Report for {name} sent to doctor! ✅")
                else:
                    st.error("Email failed. Verify Secrets settings.")
