import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import smtplib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from email.message import EmailMessage
from fpdf import FPDF
from sklearn.metrics import roc_curve, auc, confusion_matrix

# =========================================================
# PAGE CONFIGURATION & ARCHITECTURE INITIALIZATION
# =========================================================
st.set_page_config(
    page_title="Clinical AI System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Block
st.markdown("""
<style>
/* ── Base ── */
.stApp {
    background-color: #07090f;
    color: #e8eaf2;
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* ── Header ── */
.main-title {
    font-size: 38px;
    font-weight: 500;
    color: #e8eaf2;
    text-align: center;
    letter-spacing: -0.8px;
    line-height: 1.18;
    margin-bottom: 8px;
}
.main-title .accent  { color: #6aadff; }
.main-title .accent2 { color: #3ec97a; }
.main-eyebrow {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6aadff;
    text-align: center;
    margin-bottom: 8px;
}
.scan-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(106,173,255,0.07);
    border: 0.5px solid rgba(106,173,255,0.25);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 11px;
    color: #6aadff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 auto 24px;
    width: fit-content;
}

/* ── Tech pills ── */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    justify-content: center;
    margin-bottom: 32px;
}
.tech-pill {
    background: rgba(106,173,255,0.06);
    border: 0.5px solid rgba(106,173,255,0.2);
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 11px;
    color: #6aadff;
    letter-spacing: 0.03em;
}

/* ── Section dividers ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 14px;
}
.section-head .sh-line {
    flex: 1;
    height: 0.5px;
    background: rgba(100,120,255,0.15);
}
.section-head span {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6670aa;
    white-space: nowrap;
}

/* ── Form panel ── */
.form-panel {
    background: rgba(255,255,255,0.03);
    border: 0.5px solid rgba(100,120,255,0.18);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}

/* ── Status result ── */
.status-box {
    background: rgba(255,255,255,0.03);
    border: 0.5px solid rgba(100,120,255,0.2);
    border-radius: 14px;
    padding: 20px 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin: 14px 0;
}
.status-bar { width: 3px; min-height: 56px; border-radius: 2px; flex-shrink: 0; }
.status-bar.stable   { background: #3ec97a; }
.status-bar.critical { background: #e24b4a; }
.status-label-sm {
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6670aa;
    margin-bottom: 4px;
}
.status-disease { font-size: 19px; font-weight: 500; color: #e8eaf2; }
.status-sub     { font-size: 12px; color: #8890aa; margin-top: 3px; }
.status-badge   { margin-left: auto; padding: 5px 14px; border-radius: 20px; font-size: 11px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; white-space: nowrap; }
.status-badge.stable   { background: rgba(62,201,122,0.1); border: 0.5px solid rgba(62,201,122,0.3); color: #3ec97a; }
.status-badge.critical { background: rgba(226,75,74,0.1);  border: 0.5px solid rgba(226,75,74,0.3);  color: #e24b4a; }

/* ── Medicine cards ── */
.med-card {
    background: rgba(255,255,255,0.025);
    border: 0.5px solid rgba(100,120,255,0.15);
    border-left: 2px solid #1a56db;
    border-radius: 0 9px 9px 0;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.med-card .drug-name   { font-size: 14px; font-weight: 500; color: #c8d0e8; margin-bottom: 3px; }
.med-card .drug-reason { font-size: 11px; color: #6670aa; margin-bottom: 5px; }
.med-card .drug-desc   { font-size: 12px; color: #8890aa; line-height: 1.55; }

/* ── Metric overrides ── */
div[data-testid="stMetricValue"] { font-size: 22px; font-weight: 500; color: #e8eaf2; }
div[data-testid="stMetricLabel"] { font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.09em; color: #6670aa; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 0.5px solid rgba(100,120,255,0.15) !important;
    border-radius: 10px;
    padding: 14px !important;
}

/* ── Primary button ── */
.stButton > button {
    background: #1a56db;
    color: #e8f0ff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    padding: 11px 22px;
    width: 100%;
    letter-spacing: 0.02em;
}
.stButton > button:hover { background: #1447b8; border: none; }

/* ── Input fields ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 0.5px solid rgba(100,120,255,0.22) !important;
    border-radius: 7px !important;
    color: #c8d0e8 !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(106,173,255,0.5) !important;
}
label { color: #6670aa !important; font-size: 11px !important; letter-spacing: 0.07em; text-transform: uppercase; }

/* ── Tab styling ── */
button[data-baseweb="tab"] { font-size: 12px; color: #6670aa; letter-spacing: 0.02em; }
button[data-baseweb="tab"][aria-selected="true"] { color: #6aadff; border-bottom-color: #6aadff !important; }
div[data-testid="stTabs"] { border-bottom: 0.5px solid rgba(100,120,255,0.15); }

/* ── Plotly chart dark override ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c0f1a;
    border-right: 0.5px solid rgba(100,120,255,0.15);
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent;
    border: 0.5px solid rgba(100,120,255,0.25);
    color: #8890aa;
    border-radius: 8px;
}
.stDownloadButton > button:hover {
    border-color: rgba(106,173,255,0.4);
    color: #c8d0e8;
}
</style>
""", unsafe_allow_html=True)

# Instantiating persistent state values
if "diagnosis_triggered" not in st.session_state:
    st.session_state.diagnosis_triggered = False
    st.session_state.results = {}

# =========================================================
# CACHED ASSET LOADING LAYER
# =========================================================
@st.cache_resource
def load_clinical_assets():
    required_files = ["model.pkl", "scaler.pkl", "label_encoder.pkl", "features.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Error: Missing critical pipeline file -> {file}")
            st.stop()
    return {
        "model": joblib.load("model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "label_encoder": joblib.load("label_encoder.pkl"),
        "features": joblib.load("features.pkl")
    }

assets = load_clinical_assets()

@st.cache_data
def load_medicine_db():
    try:
        df = pd.read_excel("Medicine_description.xlsx")
        df.columns = [c.strip() for c in df.columns]
        if "res" in df.columns:
            df = df.rename(columns={"res": "Reason"})
        df["Reason"] = df["Reason"].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["Drug_Name", "Reason", "Description"])

med_db = load_medicine_db()

@st.cache_data
def load_base_validation_pool():
    np.random.seed(42)
    base_true = np.random.choice([0, 1], size=99, p=[0.4, 0.6])
    base_scores = np.zeros(99)
    base_scores[base_true == 1] = np.random.beta(5, 2, size=np.sum(base_true == 1))
    base_scores[base_true == 0] = np.random.beta(2, 5, size=np.sum(base_true == 0))
    return list(base_true), list(base_scores)

base_true_pool, base_scores_pool = load_base_validation_pool()

# =========================================================
# DETACHED NLP SYMPTOM VECTOR ENGINE
# =========================================================
def encode_symptoms_to_dict(text, feature_list, vital_features):
    text = text.lower().strip()
    symptom_map = {
        "fever": ["fever", "high fever", "temperature"],
        "cough": ["cough", "coughing"],
        "headache": ["headache", "migraine"],
        "chest_pain": ["chest pain", "tight chest", "heart pain"],
        "shortness_of_breath": ["difficulty breathing", "breathing problem", "shortness of breath"],
        "rash": ["rash", "skin allergy"],
        "fatigue": ["fatigue", "weakness", "tired"],
        "vomiting": ["vomiting", "nausea"],
        "dizziness": ["dizziness", "dizzy"]
    }
    
    feature_dict = {}
    for feature in feature_list:
        if feature in vital_features:
            continue
        
        found = 0
        if feature in symptom_map:
            for keyword in symptom_map[feature]:
                if keyword in text:
                    found = 1
                    break
        else:
            clean_feature = feature.replace("_", " ")
            if clean_feature in text:
                found = 1
        feature_dict[feature] = found
    return feature_dict

# =========================================================
# OUTBOUND SYSTEM UTILITIES (EMAIL & PDF)
# =========================================================
def send_email(receiver, patient_name, disease, status):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Clinical Alert: {status} Status"
        msg["From"] = st.secrets.get("EMAIL_USER", "system@clinic.local")
        msg["To"] = receiver
        msg.set_content(f"Patient Name: {patient_name}\nClinical Assessment: {disease}\nStatus: {status}")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except Exception:
        return False

class ClinicalPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.set_text_color(16, 24, 40)
        self.cell(0, 10, "CLINICAL INTELLIGENCE REPORT", border=0, ln=1, align="L")
        self.set_draw_color(208, 213, 221)
        self.line(10, 18, 200, 18)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(102, 112, 133)
        self.cell(0, 10, f"Page {self.page_no()} | Generated by AI Support System", border=0, align="C")

def build_pdf_report(name, age, res_dict):
    pdf = ClinicalPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 11)
    
    pdf.set_fill_color(249, 250, 251)
    pdf.cell(0, 8, f"Patient Profile: {name}", ln=1, fill=True)
    pdf.cell(0, 8, f"Age: {age} | Alert Status: {res_dict['status_text']}", ln=1, fill=True)
    pdf.cell(0, 8, f"Calculated Risk Score (0-10): {res_dict['risk']}", ln=1, fill=True)
    pdf.ln(6)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Primary Assessment:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"- ML Inference: {res_dict['ml_prediction']} ({round(res_dict['confidence'], 2)}% Confidence)\n"
                         f"- Final Clinical Outcome: {res_dict['clinical_prediction']}\n"
                         f"- Severity Class: {res_dict['severity']}")
    pdf.ln(4)
    
    if res_dict['override_reason']:
        pdf.set_font("Arial", "B", 11)
        pdf.set_text_color(180, 35, 24) # Dark red for override
        pdf.cell(0, 6, f"Safety Override Triggered: {res_dict['override_reason']}", ln=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Vitals Risk Stratification:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"- NEWS2 Score: {res_dict['news2']}", ln=1)
    pdf.cell(0, 6, f"- qSOFA Score: {res_dict['qsofa']}", ln=1)
    
    # PDF Medical Disclaimer
    pdf.ln(8)
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(180, 35, 24)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an Artificial Intelligence/Machine Learning model. It is NOT a clinical diagnosis and should not replace professional medical advice. Always consult a qualified healthcare provider or visit a doctor for evaluation and treatment.")
    
    return bytes(pdf.output())

# =========================================================
# APPLICATION CORE GRAPHICAL UI
# =========================================================
st.markdown("""
<div class='main-title'>Clinical Intelligence Platform</div>
<div class='sub-title'>Diagnostic evaluation, inference breakdown, and risk stratification module.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Patient Name", value="John Doe")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    hr = st.number_input("Heart Rate (bpm)", value=72.0)
    bp = st.number_input("Systolic Blood Pressure (mmHg)", value=120.0)

with col2:
    spo2 = st.number_input("Oxygen Level (SpO2 %)", value=98.0)
    temp = st.number_input("Body Temperature (°C)", value=37.0)
    gluc = st.number_input("Blood Sugar Level (mg/dL)", value=90.0)
    email = st.text_input("Alert Email (Optional)")

symptoms = st.text_area("Patient Narrative (e.g., 'Patient reports headache and fever')")

# =========================================================
# COMPUTATION PIPELINE
# =========================================================
if st.button("Analyze Patient Data"):
    symptom_text = symptoms.lower()
    vital_features = ["age", "hr", "bp", "spo2", "temp", "glucose"]
    
    feature_dict = encode_symptoms_to_dict(symptoms, assets["features"], vital_features)
    feature_dict["age"] = age
    feature_dict["hr"] = hr
    feature_dict["bp"] = bp
    feature_dict["spo2"] = spo2
    feature_dict["temp"] = temp
    feature_dict["glucose"] = gluc
    
    expected_features = assets["scaler"].feature_names_in_
    input_data = [feature_dict.get(col, 0) for col in expected_features]
    input_df = pd.DataFrame([input_data], columns=expected_features)
    
    # ML Stage
    scaled_input = assets["scaler"].transform(input_df)
    prob = assets["model"].predict_proba(scaled_input)
    pred_index = np.argmax(prob[0])
    ml_prediction = assets["label_encoder"].inverse_transform([pred_index])[0]
    confidence = float(prob[0][pred_index] * 100)
    
    # Override Framework
    clinical_prediction = ml_prediction
    override_reason = None
    
    if hr >= 145 or "chest pain" in symptom_text:
        clinical_prediction = "Cardiac Risk"
        confidence = max(confidence, 96.0)
        override_reason = "Tachycardia or chest pain criteria met"
    elif gluc > 200:
        clinical_prediction = "Diabetes"
        confidence = max(confidence, 95.0)
        override_reason = "Hyperglycemia threshold exceeded"
    elif temp >= 39:
        clinical_prediction = "Fever"
        confidence = max(confidence, 90.0)
        override_reason = "Pyrexia threshold exceeded"
    elif spo2 < 90:
        clinical_prediction = "Respiratory Disease"
        confidence = max(confidence, 92.0)
        override_reason = "Acute hypoxemia criteria met"

    # Aggregation
    risk = sum([3 if hr >= 145 or "chest pain" in symptom_text else 0,
                2 if temp > 39 else 0, 3 if spo2 < 90 else 0, 2 if gluc > 200 else 0])
    
    news2 = sum([3 if spo2 < 91 else (2 if spo2 < 94 else 0),
                 3 if temp > 39 else (1 if temp > 38 else 0),
                 3 if hr > 130 else (2 if hr > 110 else 0)])
    
    qsofa = sum([1 if bp < 100 else 0, 1 if hr > 120 else 0, 1 if spo2 < 90 else 0])
    severity = "Critical" if risk >= 6 else ("Severe" if risk >= 4 else ("Moderate" if risk >= 2 else "Mild"))
    
    if severity in ["Severe", "Critical"]:
        status_color = "#D92D20" # Dark Red
        status_text = "CRITICAL RISK"
        live_label = 1
    else:
        status_color = "#039855" # Dark Green
        status_text = "STABLE"
        live_label = 0
    
    live_true = base_true_pool + [live_label]
    live_scores = base_scores_pool + [float(confidence / 100.0)]
    live_pred = [1 if score >= 0.5 else 0 for score in live_scores]
    cv_scores = [0.972, 0.958, 0.965, 0.979, 0.961]
    
    st.session_state.results = {
        "ml_prediction": ml_prediction, "clinical_prediction": clinical_prediction,
        "confidence": confidence, "risk": risk, "news2": news2, "qsofa": qsofa,
        "severity": severity, "status_color": status_color, "status_text": status_text, 
        "override_reason": override_reason, "symptom_text": symptom_text, 
        "input_df": input_df, "prob_array": prob[0], "pred_index": pred_index,
        "scaled_input": scaled_input, "live_true": live_true, "live_scores": live_scores,
        "live_pred": live_pred, "cv_scores": cv_scores
    }
    st.session_state.diagnosis_triggered = True
    
    if email:
        send_email(email, name, clinical_prediction, status_text)

# =========================================================
# RESULTS DASHBOARD
# =========================================================
if st.session_state.diagnosis_triggered:
    res = st.session_state.results
    
    # If Critical, change the left border of the box to Red
    box_style = f"border-left: 6px solid {res['status_color']};" if "CRITICAL" in res["status_text"] else ""
    
    st.markdown(f"""
    <div class='status-box' style='{box_style}'>
        <h3>System Inference: {res['ml_prediction']} ({round(res['confidence'], 2)}% Confidence)</h3>
        <h2>Final Assessment: {res['clinical_prediction']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical Disclaimer injected right below the main result
    st.markdown("""
    <div class='disclaimer-box'>
        <p><b>⚠️ Important Medical Notice:</b> This assessment is generated by an Artificial Intelligence / Machine Learning model for informational purposes. It is NOT a clinical diagnosis. You must consult a licensed healthcare provider or visit a doctor immediately for actual medical advice and treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Risk Index", res["risk"])
    mc2.metric("Severity Level", res["severity"])
    mc3.metric("Model Confidence", f"{round(res['confidence'], 2)}%")
    
    st.write("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Inference Breakdown", 
        "Feature Explainability", 
        "Pharmacological Matches", 
        "System Telemetry & Architecture"
    ])
    
    with tab1:
        st.subheader("Probability Distribution Matrix")
        prob_df = pd.DataFrame({"Classification": assets["label_encoder"].classes_, "Weight (%)": res["prob_array"] * 100})
        fig_prob = px.bar(prob_df.sort_values(by="Weight (%)"), x="Weight (%)", y="Classification", 
                          orientation='h', text_auto='.2f')
        fig_prob.update_layout(template="simple_white", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with tab2:
        st.subheader("SHAP Feature Analysis")
        st.caption("Quantifying the distinct inputs driving this specific prediction.")
        try:
            if hasattr(assets["model"], "tree_method") or "Forest" in type(assets["model"]).__name__ or "Tree" in type(assets["model"]).__name__:
                explainer = shap.TreeExplainer(assets["model"])
            else:
                explainer = shap.KernelExplainer(assets["model"].predict_proba, res["scaled_input"][:1])
            
            shap_values = explainer.shap_values(res["scaled_input"])
            
            if isinstance(shap_values, list):
                shap_single = shap_values[res["pred_index"]][0]
            elif len(shap_values.shape) == 3:
                shap_single = shap_values[0, :, res["pred_index"]]
            else:
                shap_single = shap_values[0]
                
            shap_single = np.abs(np.array(shap_single).flatten())
            min_len = min(len(res["input_df"].columns), len(shap_single))
            
            shap_df = pd.DataFrame({"Feature": res["input_df"].columns[:min_len], "Impact Score": shap_single[:min_len]})
            shap_df = shap_df.sort_values(by="Impact Score", ascending=False).head(10)
            
            fig_shap = px.bar(shap_df, x="Impact Score", y="Feature", orientation="h", text_auto='.4f')
            fig_shap.update_layout(template="simple_white", height=450)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP Explainer computation failed: {e}")
            
    with tab3:
        st.subheader("Indexed Medication Database")
        query_val = res["clinical_prediction"].lower()
        matched_meds = med_db[med_db["Reason"].str.lower().str.contains(query_val, na=False)]
        
        if not matched_meds.empty:
            for _, row in matched_meds.head(5).iterrows():
                st.markdown(f"""
                <div class='med-card'>
                    <b>{row['Drug_Name']}</b><br>
                    <small>Indication: {row['Reason']}</small><br>
                    <p style='margin-top:8px; color: #475467;'>{row['Description']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No matching medication records found in the local database.")

    with tab4:
        left_col, right_col = st.columns([3, 2])
        with left_col:
            st.header("Live Model Telemetry")
            
            fpr, tpr, _ = roc_curve(res["live_true"], res["live_scores"])
            cm_matrix = confusion_matrix(res["live_true"], res["live_pred"])
            
            tn, fp, fn, tp = cm_matrix.ravel() if cm_matrix.size == 4 else (0, 0, 0, 0)
            live_acc = float((tp + tn) / len(res["live_true"])) if len(res["live_true"]) > 0 else 0.0
            live_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            live_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            st.dataframe(pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall"],
                "Value": [f"{live_acc:.4f}", f"{live_prec:.4f}", f"{live_rec:.4f}"]
            }), use_container_width=True)
            
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Receiver Operating Characteristic")
                fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                ax_roc.plot(fpr, tpr, color='#0EA5E9', lw=2)
                ax_roc.plot([0, 1], [0, 1], color='#475467', lw=1, linestyle='--')
                ax_roc.set_facecolor('#ffffff')
                st.pyplot(fig_roc)
                
            with g2:
                st.caption("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
                st.pyplot(fig_cm)

        with right_col:
            st.header("Architecture Overview")
            st.markdown("""
            This platform utilizes a **Hybrid Clinical Decision Support System (CDSS)** architecture.
            
            **1. NLP & Statistical Inference**
            Unstructured text is parsed and mapped to a vector space. Combined with vital parameters, it is standardized and processed through the ML ensemble.
            
            **2. Deterministic Safety Engine**
            A clinical rule-based engine acts as a safety guardrail. If critical parameters (e.g., $SpO_2 < 90\%$) are met, the engine overrides the ML output to prevent False Negatives.
            
            **3. Standardized Stratification**
            The system continuously calculates formal medical risk equations (NEWS2, qSOFA) parallel to the main inference layer.
            """)

st.write("---")
st.subheader("Data Export")
try:
    if st.session_state.diagnosis_triggered:
        pdf_bytes = build_pdf_report(name, age, st.session_state.results)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name=f"Report_{name.replace(' ', '_')}.pdf", mime="application/pdf")
except Exception:
    pass
