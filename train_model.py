import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------- LOAD YOUR FILE ----------------
df = pd.read_csv("Preprocessed_Disease_Symptoms.csv")

# ---------------- CHECK ----------------
print(df.head())

# ---------------- TARGET ----------------
y = df['Disease']
X = df.drop(columns=['Disease'])

# ---------------- ADD VITALS ----------------
np.random.seed(42)

X['age'] = np.random.randint(20, 70, len(X))
X['hr'] = np.random.randint(60, 120, len(X))
X['bp'] = np.random.randint(90, 160, len(X))
X['spo2'] = np.random.randint(85, 100, len(X))
X['temp'] = np.random.uniform(36, 40, len(X))
X['glucose'] = np.random.randint(70, 180, len(X))

# ---------------- ENCODE LABEL ----------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ---------------- SAVE ----------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model trained using YOUR dataset")

