import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("BANK LOAN.csv")
df.drop(columns = ['SN'],inplace = True)

print("Data Loaded:")
print(df.head())

# -----------------------------
# 2. Features & Target
# -----------------------------
X = df.drop("DEFAULTER", axis=1)
y = df["DEFAULTER"]

# -----------------------------
# 3. Train-Test Split (good practice)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)

# -----------------------------
# 5. Accuracy (optional but nice for class)
# -----------------------------
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# 6. Save Model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")