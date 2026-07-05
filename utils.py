"""
Shared utilities for model loading and data preprocessing.
Used by app.py, gui.py, and professional_gui.py to avoid code duplication.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")


def load_model():
    """Load the trained model from disk"""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    return joblib.load(model_path)


def load_scaler():
    """Load the fitted scaler from disk"""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run train_model.py first."
        )
    return joblib.load(scaler_path)


def load_all_models():
    """Load all trained models from disk"""
    all_models_path = os.path.join(MODEL_DIR, "all_models.pkl")
    if not os.path.exists(all_models_path):
        raise FileNotFoundError(
            f"All models not found at {all_models_path}. Run train_model.py first."
        )
    return joblib.load(all_models_path)


# Feature names in the order expected by the model
FEATURE_ORDER = [
    "male", "age", "cigsPerDay", "BPMeds", "prevalentStroke",
    "prevalentHyp", "diabetes", "totChol", "sysBP", "BMI",
    "heartRate", "glucose"
]

# Columns that need standard scaling
COLS_TO_STANDARDISE = [
    "age", "cigsPerDay", "totChol", "sysBP", "BMI", "heartRate", "glucose"
]


def prepare_sample(inputs: dict, scaler) -> np.ndarray:
    """
    Convert a dict of user inputs to a scaled numpy array for prediction.

    Parameters
    ----------
    inputs : dict
        Mapping of feature name -> raw user value.
    scaler : StandardScaler
        The fitted scaler used during training.

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    row = {col: float(inputs.get(col, 0)) for col in FEATURE_ORDER}
    raw_df = pd.DataFrame([row])
    raw_df[COLS_TO_STANDARDISE] = scaler.transform(raw_df[COLS_TO_STANDARDISE])
    return raw_df[FEATURE_ORDER].values


def evaluate_accuracies(all_models: dict) -> dict:
    """
    Evaluate saved models on the test split and return accuracy dict.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "framingham.csv"))
    df1 = df.drop(columns=["education"])
    df2 = df1.drop(columns=["currentSmoker", "diaBP"])
    imputer = SimpleImputer(strategy="most_frequent")
    df3 = pd.DataFrame(
        imputer.fit_transform(df2), columns=df2.columns, index=df2.index
    )
    df3 = df3[~(df3["sysBP"] > 220)]
    df3 = df3[~(df3["BMI"] > 43)]
    df3 = df3[~(df3["heartRate"] > 125)]
    df3 = df3[~(df3["glucose"] > 200)]
    df3 = df3[~(df3["totChol"] > 450)]

    sc = StandardScaler()
    df3[COLS_TO_STANDARDISE] = sc.fit_transform(df3[COLS_TO_STANDARDISE])

    X = df3.drop(columns=["TenYearCHD"])
    Y = df3["TenYearCHD"]
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

    accs = {}
    for name, model in all_models.items():
        accs[name] = round(accuracy_score(Y_test, model.predict(X_test)) * 100, 2)
    return accs
