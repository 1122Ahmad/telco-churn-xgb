# ================================================
# preprocessing.py
# ================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path):
    df = pd.read_csv(path)

    # Encode target
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Fix TotalCharges blank → NaN → numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Split
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train):
    num_features = selector(dtype_include=np.number)(X_train)
    cat_features = selector(dtype_include=object)(X_train)

    numeric_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_features),
            ("cat", categorical_pipe, cat_features),
        ],
        remainder="drop"
    )
    return preprocessor