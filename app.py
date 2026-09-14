import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_trained_pipeline():
    data_path = BASE_DIR / "public_transport_delays.csv"
    df = pd.read_csv(data_path)
    
    # Replace 'delay' below with your target column if named differently (e.g., 'Delayed' or 'target')
    target_col = 'delay' if 'delay' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Dynamically separate numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    full_pipeline.fit(X, y)
    return full_pipeline, X.columns

model_pipeline, feature_names = load_trained_pipeline()
st.success("Transit Delay Predictor Model Loaded Successfully!")
