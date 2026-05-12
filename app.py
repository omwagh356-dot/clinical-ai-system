import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Intelligent Clinical Decision Support System",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
    color:white;
}

.main-title{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.status-box{
    padding:25px;
    border-radius:18px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
    margin-top:20px;
    margin-bottom:20px;
}

.med-card{
    background:rgba(0,0,0,0.35);
    padding:15px;
    border-radius:12px;
    )
