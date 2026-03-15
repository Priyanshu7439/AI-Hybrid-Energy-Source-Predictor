import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
from pathlib import Path

from src.optimization.hybrid_optimizer import optimize_energy
from src.llm_agent.energy_advisor import explain_energy


st.set_page_config(page_title="AI Hybrid Energy Platform", layout="wide")

st.title("⚡ AI Hybrid Energy Platform")


ARTIFACT_PATH = Path("artifacts")

solar_model = joblib.load(ARTIFACT_PATH / "solar_model.pkl")
wind_model = joblib.load(ARTIFACT_PATH / "wind_model.pkl")


col1,col2 = st.columns(2)


with col1:

    st.header("Solar Inputs")

    irradiation = st.slider("Irradiation",0,1500,800)
    temp = st.slider("Temperature",-10,60,30)
    module = st.slider("Module Temp",-10,80,35)

    hour = st.slider("Hour",0,23,12)
    day = st.slider("Day",1,31,15)
    month = st.slider("Month",1,12,6)


with col2:

    st.header("Wind Inputs")

    wind_speed = st.slider("Wind Speed",0,25,6)
    direction = st.slider("Direction",0,360,250)
    theoretical = st.slider("Theoretical Power",0,2000,700)



if st.button("Predict Energy"):

    solar_input = np.array([[irradiation, temp, module, hour, day, month]])
    wind_input = np.array([[wind_speed, direction, theoretical]])

    response = requests.get("http://127.0.0.1:8000/predict")
    data = response.json()

    solar = data["solar_power"]
    wind = data["wind_power"]
    total = data["total_energy"]
    source = data["recommended_source"]

    st.metric("Solar Power", round(solar,2))
    st.metric("Wind Power", round(wind,2))
    st.metric("Total Energy", round(total,2))

    st.success(f"Recommended Source: {source}")


    chart_data = {
        "source":["solar","wind"],
        "energy":[solar,wind]
    }

    fig = px.bar(chart_data,x="source",y="energy")

    st.plotly_chart(fig)



    ratio = optimize_energy(solar,wind)

    st.subheader("Hybrid Optimization")

    st.write(ratio)



    explanation = explain_energy(ratio["recommended_source"])

    st.subheader("AI Advisor")

    st.write(explanation)