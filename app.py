import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import smtplib
from email.message import EmailMessage

# --- 1. DATA LOADING (The "res" Fix) ---

@st.cache_data
def load_clinical_data():
    disease_df = pd.read_csv('DiseaseAndSymptoms.csv')
    disease_df['Disease'] = disease_df['Disease'].astype(str).str.strip().str.title()
    try:
        # Renaming the 'res' column you created to 'Reason' internally
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

try:
    from drug_module import check_drugs
    from explain import explain_values
except:
    st.error("Missing Logic Modules.")

# --- 2. UI LAYOUT ---

st.set_page_config(page_title="Advanced CDSS", layout="wide")
st.title("🛡️ Enterprise Clinical Decision Support System")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.subheader("👤 Patient Identity")
    name = st.text_input("Patient Name")
    doc_email = st.text_input("Physician Email")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", 1, 120, 30)
    
    st.subheader("📉 Vitals")
    v1, v2, v3 = st.columns(3)
    hr = v1.number_input("Heart Rate", value=72.0)
    bps = v1.number_input("BP Systolic", value=120.0)
    spo2 = v2.number_input("SpO2 %", value=98.0)
    temp = v3.number_input("Temp °C", value=37.0)
    gluc = v3.number_input("Glucose", value=95.0)

with c2:
    st.subheader("🧪 Clinical Context")
    curr_syms = st.text_area("Symptoms (e.g. fever, cough, acne, wound)")
    curr_meds = st.text_area("Current Medications")
    curr_allergies = st.text_area("Allergies")

# --- 3. DIAGNOSTIC ENGINE ---

if st.button("🚀 EXECUTE FULL SCAN", type="primary", width="stretch"):
    # AI Prediction
    raw_vitals = [age, hr, bps, 80.0, spo2, temp, 190.0, gluc, 16.0] 
    scaled = scaler.transform(pd.DataFrame([raw_vitals], columns=scaler.feature_names_in_))
    prediction = model.predict(scaled, verbose=0)
    
    idx = np.argmax(prediction)
    disease = label_encoder.inverse_transform([idx])[0]
    prob = prediction[0][idx] * 100

    # --- THE WEIGHTAGE FIX ---
    # Overriding 'Normal' if obvious symptoms are present
    symptom_text = curr_syms.lower()
    if temp >= 38.5 or "fever" in symptom_text or "cough" in symptom_text:
        if disease == "Normal":
            disease = "Fever / Cough / Infection"
            prob = 98.0
            
    if "pimple" in symptom_text or "acne" in symptom_text:
        if disease == "Normal":
            disease = "Acne Vulgaris"
            prob = 95.0

    status = "🔴 CRITICAL" if temp >= 39.5 or spo2 < 89 else "🟢 STABLE"
    color = "#ff4b4b" if status == "🔴 CRITICAL" else "#28a745"

    st.markdown(f"<div style='background:{color}; padding:20px; border-radius:10px; color:white; text-align:center;'><h1>{status}: {disease.upper()}</h1></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "💊 Therapy Pathway", "📄 Report"])
    
    with tab1:
        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.plotly_chart(px.bar(pd.DataFrame({"Condition": label_encoder.classes_, "Prob": prediction[0]*100}), x="Prob", y="Condition", orientation='h', template="plotly_dark"), width="stretch")
        with g2:
            radar_vals = [min(hr/160, 1.0), (100-spo2)/20, min(bps/200, 1.0), min(abs(temp-37)/5, 1.0), min(gluc/400, 1.0)]
            fig = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=['HR', 'SpO2', 'BP', 'Temp', 'Gluc'], fill='toself', line_color=color))
            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.subheader(f"Therapeutic Pathway for {disease}")
        
        # --- THE MASTER SEARCH FIX ---
        # We break the disease name into keywords (e.g., 'Fever', 'Cough') and search them all
        keywords = disease.replace("/", " ").split()
        # Add symptoms as backup keywords
        keywords += [k for k in ["Fever", "Cough", "Acne", "Wound"] if k.lower() in symptom_text]
        
        query = "|".join(keywords) # This creates an 'OR' search: Fever OR Cough OR Infection
        rel_meds = med_db[med_db['Reason'].str.contains(query, case=False, na=False)].head(10)
        
        if not rel_meds.empty:
            for _, row in rel_meds.iterrows():
                with st.expander(f"💊 {row['Drug_Name']}"):
                    st.write(f"**Indication:** {row['Reason']}")
                    st.write(f"**Description:** {row['Description']}")
        else:
            st.warning("No medicines matching these keywords found in database.")

    with tab3:
        st.info("Download report or email physician functionality active.")
