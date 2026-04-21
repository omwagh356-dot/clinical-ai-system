#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the preprocessed data
df = pd.read_csv('Preprocessed_Disease_Symptoms.csv')

# 2. Separate Features (X) and Target (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# 3. Encode the Target Labels (convert disease names to numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Train the Model
# Random Forest is ideal for this dataset as it handles binary (0/1) features perfectly
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Export the trained assets
# We save the feature names (symptoms) to ensure the app uses the same order as the model
joblib.dump(model, 'symptom_model.pkl')
joblib.dump(le, 'symptom_encoder.pkl')
joblib.dump(X.columns.tolist(), 'symptom_features.pkl')

print("Assets saved: symptom_model.pkl, symptom_encoder.pkl, symptom_features.pkl")


# In[ ]:




