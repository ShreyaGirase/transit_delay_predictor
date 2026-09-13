import streamlit as st
import pandas as pd
import joblib

model_pipeline = joblib.load("models/transit_model.pkl")

st.title("Transit Delay Predictor")

temperature_C = st.number_input("Temperature (°C)", value=22.5)
traffic_congestion_index = st.number_input("Traffic Congestion Index", value=7.5)
hour = st.slider("Hour of Day", 0, 23, 17)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
event_attendance_est = st.number_input("Estimated Event Attendance", value=5000.0)
transport_type = st.selectbox("Transport Type", ["Bus", "Train", "Tram"])
weather_condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
event_type = st.selectbox("Event Type", ["None", "Concert", "Sports", "Festival"])
route_id = st.text_input("Route ID", value="R01")

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
