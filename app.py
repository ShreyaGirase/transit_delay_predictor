import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Transit Delay Predictor", layout="centered")
st.title("🚌 Public Transit Delay Predictor")
st.write("Enter the route details below to estimate the exact delay time in minutes.")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_trained_pipeline():
    data_path = BASE_DIR / "public_transport_delays.csv"
    df = pd.read_csv(data_path)
    
    # Target column (delay duration in minutes)
    target_col = 'delay' if 'delay' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
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

    # Regressor outputs exact continuous values
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    full_pipeline.fit(X, y)
    return full_pipeline, X, numeric_features, categorical_features

model_pipeline, X_df, num_cols, cat_cols = load_trained_pipeline()

# --- UI INPUT FORM ---
st.subheader("Route Parameters")
input_data = {}

for col in num_cols:
    min_val = float(X_df[col].min())
    max_val = float(X_df[col].max())
    mean_val = float(X_df[col].mean())
    input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

for col in cat_cols:
    unique_opts = X_df[col].dropna().unique().tolist()
    input_data[col] = st.selectbox(f"{col}", options=unique_opts)

# Prediction execution
if st.button("Predict Delay", type="primary"):
    input_df = pd.DataFrame([input_data])
    predicted_delay = model_pipeline.predict(input_df)[0]
    
    st.markdown("---")
    # Display precise numerical output formatted to 1 decimal place
    if predicted_delay > 0:
        st.metric(label="Estimated Delay Duration", value=f"{predicted_delay:.1f} mins")
    else:
        st.metric(label="Estimated Delay Duration", value="0.0 mins (On Time)")
