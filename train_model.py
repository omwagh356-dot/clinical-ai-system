#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_excel("clinical_dataset.xlsx")

# =====================================================
# TARGET COLUMN
# =====================================================

TARGET = "disease"

# =====================================================
# FEATURES + TARGET
# =====================================================

X = df.drop(TARGET, axis=1)

y = df[TARGET]

# =====================================================
# SAVE FEATURE NAMES
# =====================================================

joblib.dump(
    list(X.columns),
    "features.pkl"
)

# =====================================================
# LABEL ENCODER
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# IMPORTANT
print("\nCLASS ORDER:\n")

for i, c in enumerate(label_encoder.classes_):
    print(i, "->", c)

# SAVE ENCODER
joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================================
# SCALER
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# SAVE SCALER
joblib.dump(
    scaler,
    "scaler.pkl"
)

# =====================================================
# MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# =====================================================
# EVALUATION
# =====================================================

pred = model.predict(X_test_scaled)

acc = accuracy_score(
    y_test,
    pred
)

print("\nMODEL ACCURACY:", acc)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "model.pkl"
)

print("\n✅ MODEL TRAINED SUCCESSFULLY")


# In[ ]:




