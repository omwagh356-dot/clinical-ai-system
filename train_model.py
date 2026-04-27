{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7c1d88e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import joblib\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9489e1d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c4d9d1ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Filtered Dataset Shape: (360, 132)\n",
      "X shape: (360, 137)\n",
      "y shape: (360,)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\anaconda3\\lib\\site-packages\\xgboost\\core.py:158: UserWarning: [16:53:26] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-08cbc0333d8d4aae1-1\\xgboost\\xgboost-ci-windows\\src\\learner.cc:740: \n",
      "Parameters: { \"use_label_encoder\" } are not used.\n",
      "\n",
      "  warnings.warn(smsg, UserWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✅ Accuracy: 1.0\n",
      "\n",
      "📊 Classification Report:\n",
      "\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        24\n",
      "           1       1.00      1.00      1.00        24\n",
      "           2       1.00      1.00      1.00        24\n",
      "\n",
      "    accuracy                           1.00        72\n",
      "   macro avg       1.00      1.00      1.00        72\n",
      "weighted avg       1.00      1.00      1.00        72\n",
      "\n",
      "\n",
      "✅ Model + scaler + encoder + features saved successfully!\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# ---------------- LOAD ----------------\n",
    "df = pd.read_csv(r\"C:\\Users\\Admin\\Downloads\\Preprocessed_Disease_Symptoms.csv\")\n",
    "\n",
    "# ---------------- CLEAN ----------------\n",
    "df['Disease'] = df['Disease'].astype(str).str.strip().str.title()\n",
    "\n",
    "# ---------------- LIMIT DISEASES ----------------\n",
    "allowed = [\"Fever\", \"Respiratory\", \"Allergy\", \"Diabetes\", \"Hypertension\"]\n",
    "\n",
    "df = df[df['Disease'].isin(allowed)].reset_index(drop=True)\n",
    "\n",
    "print(\"Filtered Dataset Shape:\", df.shape)\n",
    "\n",
    "# ---------------- SPLIT FEATURES ----------------\n",
    "X = df.drop(columns=['Disease']).reset_index(drop=True)\n",
    "y = df['Disease'].reset_index(drop=True)\n",
    "\n",
    "# ---------------- ADD REALISTIC VITALS ----------------\n",
    "def generate_vitals(disease):\n",
    "    if disease == \"Fever\":\n",
    "        return [np.random.randint(20,60), 85, 120, 96, np.random.uniform(38,40), 100]\n",
    "    elif disease == \"Respiratory\":\n",
    "        return [np.random.randint(20,60), 95, 130, np.random.randint(85,92), 37.5, 100]\n",
    "    elif disease == \"Diabetes\":\n",
    "        return [np.random.randint(30,70), 85, 140, 96, 37, np.random.randint(180,300)]\n",
    "    elif disease == \"Hypertension\":\n",
    "        return [np.random.randint(30,70), 80, np.random.randint(150,180), 97, 37, 100]\n",
    "    else:  # Allergy\n",
    "        return [np.random.randint(20,60), 75, 120, 98, 37, 100]\n",
    "\n",
    "vitals = df['Disease'].apply(generate_vitals)\n",
    "\n",
    "vitals_df = pd.DataFrame(vitals.tolist(), columns=[\n",
    "    'age','hr','bp','spo2','temp','glucose'\n",
    "]).reset_index(drop=True)\n",
    "\n",
    "# ---------------- COMBINE ----------------\n",
    "X = pd.concat([X, vitals_df], axis=1)\n",
    "\n",
    "# ---------------- CHECK SHAPE ----------------\n",
    "print(\"X shape:\", X.shape)\n",
    "print(\"y shape:\", y.shape)\n",
    "\n",
    "# ---------------- ENCODE LABEL ----------------\n",
    "le = LabelEncoder()\n",
    "y_encoded = le.fit_transform(y)\n",
    "\n",
    "# ---------------- SCALE ----------------\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# ---------------- SPLIT ----------------\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42\n",
    ")\n",
    "\n",
    "# ---------------- MODEL ----------------\n",
    "model = XGBClassifier(\n",
    "    n_estimators=300,\n",
    "    max_depth=6,\n",
    "    learning_rate=0.1,\n",
    "    subsample=0.8,\n",
    "    colsample_bytree=0.8,\n",
    "    eval_metric=\"mlogloss\",\n",
    "    use_label_encoder=False\n",
    ")\n",
    "\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# ---------------- EVALUATE ----------------\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "print(\"\\n✅ Accuracy:\", accuracy)\n",
    "print(\"\\n📊 Classification Report:\\n\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# ---------------- SAVE FILES ----------------\n",
    "joblib.dump(model, \"model.pkl\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n",
    "joblib.dump(le, \"label_encoder.pkl\")\n",
    "joblib.dump(X.columns.tolist(), \"features.pkl\")\n",
    "\n",
    "print(\"\\n✅ Model + scaler + encoder + features saved successfully!\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30f2e057",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
