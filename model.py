import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES = ["age", "bmi", "glucose", "systolic_bp", "family_history_yes", "activity_level"]

def _synthetic_data(n=5000, seed=17):
    """Artificial data for software demonstration only; not clinical evidence."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 91, n)
    bmi = np.clip(rng.normal(28, 5.5, n), 15, 50)
    glucose = np.clip(rng.normal(108, 28, n), 60, 300)
    bp = np.clip(rng.normal(125, 18, n), 80, 220)
    family = rng.binomial(1, 0.28, n)
    activity = rng.choice([0, 1, 2, 3], size=n, p=[0.20, 0.35, 0.30, 0.15])
    z = (-5.0 + 0.025 * (age - 40) + 0.075 * (bmi - 25)
         + 0.035 * (glucose - 100) + 0.012 * (bp - 120)
         + 0.65 * family - 0.20 * activity)
    probability = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, probability)
    X = pd.DataFrame({
        "age": age, "bmi": bmi, "glucose": glucose, "systolic_bp": bp,
        "family_history_yes": family, "activity_level": activity
    })
    return X, y

def train_demo_model():
    X, y = _synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=220, max_depth=8, min_samples_leaf=8,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, (probabilities >= 0.5).astype(int)),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }
    return model, metrics

def _encode_patient(patient):
    activity_map = {"Very low": 0, "Low": 1, "Moderate": 2, "High": 3}
    return pd.DataFrame([{
        "age": float(patient["age"]),
        "bmi": float(patient["bmi"]),
        "glucose": float(patient["glucose"]),
        "systolic_bp": float(patient["systolic_bp"]),
        "family_history_yes": 1 if patient["family_history"] == "Yes" else 0,
        "activity_level": activity_map.get(patient["activity"], 1)
    }], columns=FEATURES)

def predict_risk(model, patient):
    score = float(model.predict_proba(_encode_patient(patient))[0, 1]) * 100
    category = "Lower demo score" if score < 30 else ("Intermediate demo score" if score < 60 else "Higher demo score")
    return {"risk": score, "category": category}

def make_demo_history(patient):
    rng = np.random.default_rng(2026)
    months = np.arange(13)
    glucose = np.clip(patient["glucose"] + np.linspace(0, rng.normal(4, 4), 13) + rng.normal(0, 3, 13), 60, 300)
    bmi = np.clip(patient["bmi"] + np.linspace(0, rng.normal(-0.4, 0.5), 13) + rng.normal(0, 0.15, 13), 15, 50)
    return pd.DataFrame({"Month": months, "Glucose (illustrative)": glucose, "BMI (illustrative)": bmi})
