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

# Features expected by the updated UI design (traffic_congestion_index removed)
FEATURE_COLS = [
    "temperature_C",
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
    
    target_col = 'delay' if 'delay' in df.columns else df.columns[-1]
    
    # Filter features to present columns
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    
    X = df[available_features].copy()
    y = df[target_col]
    
    # Extract unique route IDs for dropdown options
    unique_routes = sorted(df['route_id'].dropna().unique().tolist()) if 'route_id' in df.columns else ["R01", "R02", "R03"]
    
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
    return full_pipeline, unique_routes

# Load trained pipeline and dynamic route list
model_pipeline, route_options = load_trained_pipeline()

# --- UI INPUTS ---
temperature_C = st.number_input("Temperature (°C)", value=22.5)
hour = st.slider("Hour of Day", 0, 23, 17)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

# Event attendance in Percentage
event_attendance_pct = st.slider("Estimated Event Attendance (%)", min_value=0, max_value=100, value=50, step=1)

# Transport type with Metro included
transport_type = st.selectbox("Transport Type", ["Bus", "Train", "Tram", "Metro"])
weather_condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
event_type = st.selectbox("Event Type", ["None", "Concert", "Sports", "Festival"])

# Dropdown for Route ID
route_id = st.selectbox("Route ID", route_options)

# --- PREDICTION EXECUTION ---
if st.button("Predict Delay"):
    input_df = pd.DataFrame([{
        "temperature_C": temperature_C,
        "hour": hour,
        "day_of_week": day_of_week,
        "event_attendance_est": float(event_attendance_pct) / 100.0,
        "transport_type": transport_type,
        "weather_condition": weather_condition,
        "event_type": event_type,
        "route_id": route_id
    }])
    
    prediction = model_pipeline.predict(input_df)[0]
    probability = model_pipeline.predict_proba(input_df)[:, 1][0]
    
    st.write(f"**Delayed Prediction:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Delay Probability:** {probability:.1%}")
