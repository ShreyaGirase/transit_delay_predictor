import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Transit Delay Predictor", layout="centered")
st.title("🚌 Public Transit Delay Predictor")
st.write("Enter the route details below to predict the probability of a delay.")

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_trained_pipeline():
    data_path = BASE_DIR / "public_transport_delays.csv"
    df = pd.read_csv(data_path)
    
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

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    full_pipeline.fit(X, y)
    return full_pipeline, X, numeric_features, categorical_features

model_pipeline, X_df, num_cols, cat_cols = load_trained_pipeline()

# --- UI INPUT FORM ---
st.subheader("Route Parameters")
input_data = {}

for col in num_cols:
    min_val = int(X_df[col].min())
    max_val = int(X_df[col].max())
    mean_val = int(X_df[col].mean())
    col_lower = col.lower()

    # Enforce specific bounds and integer steps based on column names
    if 'hour' in col_lower:
        input_data[col] = st.number_input(f"{col} (0-23)", min_value=0, max_value=23, value=min(mean_val, 23), step=1)
    elif 'dayofweek' in col_lower or 'day_of_week' in col_lower:
        input_data[col] = st.number_input(f"{col} (0=Mon, 6=Sun)", min_value=0, max_value=6, value=min(mean_val, 6), step=1)
    elif 'month' in col_lower:
        input_data[col] = st.number_input(f"{col} (1-12)", min_value=1, max_value=12, value=min(mean_val, 12), step=1)
    else:
        # Standard integer input for general counts
        input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, step=1)

for col in cat_cols:
    unique_opts = X_df[col].dropna().unique().tolist()
    input_data[col] = st.selectbox(f"{col}", options=unique_opts)

# Prediction Execution
if st.button("Predict Delay Probability", type="primary"):
    input_df = pd.DataFrame([input_data])
    
    # Calculate exact probability of delay (Class 1)
    probabilities = model_pipeline.predict_proba(input_df)[0]
    delay_prob = probabilities[1] * 100 if len(probabilities) > 1 else probabilities[0] * 100
    
    st.markdown("---")
    st.metric(label="Delay Probability", value=f"{delay_prob:.1f}%")
    st.progress(int(delay_prob))

    if delay_prob >= 50:
        st.warning("⚠️ **High Risk:** High chance of delay under these route conditions.")
    else:
        st.success("✅ **Low Risk:** Route is expected to run on schedule.")
