import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Transit Delay Predictor", layout="centered")
st.title("Transit Delay Predictor")

BASE_DIR = Path(__file__).resolve().parent

# Features expected by your initial UI design
FEATURE_COLS = [
    "temperature_C",
    "traffic_congestion_index",
    "hour",
    "day_of_week",
    "event_attendance_est",
    "transport_type",
    "weather_condition",
    "event_type",
    "route_id"
]

@st.cache_resource
def load_trained_pipeline():
    data_path = BASE_DIR / "public_transport_delays.csv"
    df = pd.read_csv(data_path)
    
    # Identify target column ('delay')
    target_col = 'delay' if 'delay' in df.columns else df.columns[-1]
    
    # Use only the specific features present in your original code
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    
    X = df[available_features]
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
    return full_pipeline

# Train pipeline directly in memory at runtime
model_pipeline = load_trained_pipeline()

# --- ORIGINAL UI INPUTS ---
temperature_C = st.number_input("Temperature (°C)", value=22.5)
traffic_congestion_index = st.number_input("Traffic Congestion Index", value=7.5)
hour = st.slider("Hour of Day", 0, 23, 17)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
event_attendance_est = st.number_input("Estimated Event Attendance", value=5000.0)
transport_type = st.selectbox("Transport Type", ["Bus", "Train", "Tram"])
weather_condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
event_type = st.selectbox("Event Type", ["None", "Concert", "Sports", "Festival"])
route_id = st.text_input("Route ID", value="R01")

# --- PREDICTION EXECUTION ---
if st.button("Predict Delay"):
    input_df = pd.DataFrame([{
        "temperature_C": temperature_C,
        "traffic_congestion_index": traffic_congestion_index,
        "hour": hour,
        "day_of_week": day_of_week,
        "event_attendance_est": event_attendance_est,
        "transport_type": transport_type,
        "weather_condition": weather_condition,
        "event_type": event_type,
        "route_id": route_id
    }])
    
    prediction = model_pipeline.predict(input_df)[0]
    probability = model_pipeline.predict_proba(input_df)[:, 1][0]
    
    st.write(f"**Delayed Prediction:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Delay Probability:** {probability:.1%}")
