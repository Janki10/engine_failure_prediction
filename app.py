import streamlit as st
import pandas as pd
import numpy as np
import joblib  # if you saved model with joblib
import matplotlib.pyplot as plt
# Sidebar Menu
menu = st.sidebar.radio(
    "Select View",
    ("Engine Health Overview", "Live Sensor Trends", "Maintenance Planning")
)

# Page Title
st.title("🚀 Engine Health Monitoring Dashboard")

st.markdown("""
Welcome to the Engine Health Monitoring System.  
This dashboard shows the real-time health status of engines based on sensor data and machine learning predictions.
""")


# Load saved model and scalers
model = joblib.load('src/gbm_model.pkl')
scaler_sensors = joblib.load('src/scaler_sensors.pkl')
scaler_settings = joblib.load('src/scaler_settings.pkl')

# Load test data
test_data = pd.read_csv('data/test_FD001.txt', sep=' ', header=None)
test_data.drop([26, 27], axis=1, inplace=True)

# Assign columns
column_names = ['engine_no', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_measurement_{i}' for i in range(1, 22)]
test_data.columns = column_names

# Feature columns
setting_cols = ['setting1', 'setting2', 'setting3']
sensor_cols = [f'sensor_measurement_{i}' for i in range(1, 22)]
feature_cols = setting_cols + sensor_cols

# Apply scalers
test_data[sensor_cols] = scaler_sensors.transform(test_data[sensor_cols])
test_data[setting_cols] = scaler_settings.transform(test_data[setting_cols])

# Get latest cycle per engine
latest_test_data = test_data.groupby('engine_no').last().reset_index()

# Predict RUL
X_test = latest_test_data[feature_cols]
predicted_rul = model.predict(X_test)

# Add RUL to dataframe
latest_test_data['Predicted_RUL'] = predicted_rul

# Quick Table
st.subheader("🔎 Current Engine Health Overview")
st.dataframe(latest_test_data[['engine_no', 'cycle', 'Predicted_RUL']].style.format({'Predicted_RUL': '{:.0f}'}))

# ---------------------------- ENGINE HEALTH OVERVIEW --------------------------
if menu == "Engine Health Overview":
    st.header("🛠️ Engine Health Overview")
    st.dataframe(latest_test_data[['engine_no', 'cycle', 'Predicted_RUL']].style.format({'Predicted_RUL': '{:.0f}'}))

# ---------------------------- LIVE SENSOR TRENDS --------------------------
elif menu == "Live Sensor Trends":
    st.header("📈 Live Sensor Trends")

    # Engine selection
    engine_id = st.selectbox("Select Engine No:", latest_test_data['engine_no'].unique())

    # Get that engine's history
    engine_history = test_data[test_data['engine_no'] == engine_id]

    # Select sensor
    sensor_to_plot = st.selectbox("Select Sensor:", sensor_cols)

    # Plot
    st.line_chart(engine_history[['cycle', sensor_to_plot]].set_index('cycle'))

# ---------------------------- MAINTENANCE PLANNING --------------------------
elif menu == "Maintenance Planning":
    st.header("🧹 Maintenance Planning")

    # Risk Level based on RUL
    latest_test_data['Risk_Level'] = latest_test_data['Predicted_RUL'].apply(lambda x: "High" if x <= 50 else ("Medium" if x <= 100 else "Low"))

    # Show Priority List
    st.subheader("🔴 High Risk Engines (Needs Immediate Maintenance)")
    high_risk_engines = latest_test_data[latest_test_data['Risk_Level'] == 'High']
    st.dataframe(high_risk_engines[['engine_no', 'cycle', 'Predicted_RUL']].style.format({'Predicted_RUL': '{:.0f}'}))

    # Pie chart of risk distribution
    st.subheader("📊 Risk Distribution")
    risk_counts = latest_test_data['Risk_Level'].value_counts()
    st.pyplot(risk_counts.plot.pie(autopct='%1.1f%%', startangle=90, figsize=(5,5), title="Risk Levels").figure)
