import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_trained_pipeline():
    df = pd.read_csv("public_transport_delays.csv")

    numeric_features = ['temperature_C', 'humidity_percent', 'wind_speed_kmh',
                         'precipitation_mm', 'traffic_congestion_index',
                         'event_attendance_est', 'hour', 'day_of_week', 'is_weekend']
    categorical_features = ['transport_type', 'weather_condition', 'event_type', 'route_id']

    X = df[numeric_features + categorical_features]
    y = df['delayed']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

model_pipeline = load_trained_pipeline()

st.title("Transit Delay Predictor")

temperature_C = st.number_input("Temperature (°C)", value=22.5)
humidity_percent = st.number_input("Humidity (%)", value=60.0)
wind_speed_kmh = st.number_input("Wind Speed (km/h)", value=10.0)
precipitation_mm = st.number_input("Precipitation (mm)", value=0.0)
traffic_congestion_index = st.number_input("Traffic Congestion Index", value=7.5)
hour = st.slider("Hour of Day", 0, 23, 17)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = st.selectbox("Is Weekend?", [0, 1])
event_attendance_est = st.number_input("Estimated Event Attendance", value=5000.0)
transport_type = st.selectbox("Transport Type", ["Bus", "Train", "Tram"])
weather_condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
event_type = st.selectbox("Event Type", ["None", "Concert", "Sports", "Festival"])
route_id = st.text_input("Route ID", value="R01")

if st.button("Predict Delay"):
    input_df = pd.DataFrame([{
        "temperature_C": temperature_C,
        "humidity_percent": humidity_percent,
        "wind_speed_kmh": wind_speed_kmh,
        "precipitation_mm": precipitation_mm,
        "traffic_congestion_index": traffic_congestion_index,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
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
