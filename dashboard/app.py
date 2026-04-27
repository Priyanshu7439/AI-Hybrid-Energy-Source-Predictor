import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import streamlit as st
import numpy as np
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from src.llm_agent.energy_advisor import explain_energy

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Hybrid Energy Platform", layout="wide")

st.title("⚡ AI Hybrid Energy Platform")


def create_session_with_retries():
    """Create requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


col1, col2 = st.columns(2)

with col1:
    st.header("☀️ Solar Inputs")

    irradiation = st.slider("Irradiation (W/m²)", 0, 1500, 800)
    temp = st.slider("Temperature (°C)", -10, 60, 30)
    module = st.slider("Module Temp (°C)", -10, 80, 35)

    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)


with col2:
    st.header("💨 Wind Inputs")

    wind_speed = st.slider("Wind Speed (m/s)", 0, 25, 6)
    direction = st.slider("Direction (°)", 0, 360, 250)
    theoretical = st.slider("Theoretical Power (kW)", 0, 2000, 700)


if st.button("🔮 Predict Energy", use_container_width=True):
    
    params = {
        "irradiation": irradiation,
        "temperature": temp,
        "module": module,
        "hour": hour,
        "day": day,
        "month": month,
        "wind_speed": wind_speed,
        "direction": direction,
        "theoretical": theoretical
    }

    # Get API URL from environment variable
    API_URL = "https://ai-hybrid-energy-source-predictor-production.up.railway.app/predict"

    try:
        # Use session with retries
        session = create_session_with_retries()
        response = session.get(API_URL, params=params, timeout=10)
        
        # Check HTTP status
        if response.status_code != 200:
            try:
                error_msg = response.json().get("detail", response.text)
            except:
                error_msg = response.text
            st.error(f"❌ API Error {response.status_code}: {error_msg}")
            st.stop()
        
        # Validate JSON response
        try:
            data = response.json()
        except (requests.exceptions.JSONDecodeError, ValueError):
            st.error("❌ API returned invalid JSON. The server might be returning an error page.")
            st.stop()
        
        # Validate response structure
        required_fields = ["solar_power", "wind_power", "total_energy", "recommended_source"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            st.error(f"❌ API response missing fields: {', '.join(missing_fields)}")
            st.stop()

        # Extract and validate data types
        try:
            solar = float(data["solar_power"])
            wind = float(data["wind_power"])
            total = float(data["total_energy"])
            source = str(data["recommended_source"])
        except (ValueError, TypeError) as e:
            st.error(f"❌ Invalid data types in API response: {e}")
            st.stop()

        st.write("✅ API response:", data)

        # Display metrics
        col_solar, col_wind, col_total = st.columns(3)
        with col_solar:
            st.metric("☀️ Solar Power", f"{round(solar, 2)} kW")
        with col_wind:
            st.metric("💨 Wind Power", f"{round(wind, 2)} kW")
        with col_total:
            st.metric("⚡ Total Energy", f"{round(total, 2)} kW")

        st.success(f"✅ Recommended Source: **{source}**")

        # Display chart
        chart_data = {
            "source": ["Solar", "Wind"],
            "energy": [solar, wind]
        }

        fig = px.bar(
            chart_data,
            x="source",
            y="energy",
            title="Energy Output Comparison",
            labels={"energy": "Power (kW)", "source": "Source"},
            color="source",
            color_discrete_map={"Solar": "#FDB462", "Wind": "#80B1D3"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display optimization details
        st.subheader("📊 Hybrid Optimization")
        st.write(f"""
        - **Solar Output**: {round(solar, 2)} kW
        - **Wind Output**: {round(wind, 2)} kW
        - **Combined Output**: {round(total, 2)} kW
        - **Better Source**: {source} ({round(max(solar, wind), 2)} kW)
        """)

        # Display AI advisor explanation
        explanation = explain_energy(source)
        st.subheader("🤖 AI Energy Advisor")
        st.info(explanation)

    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Server may be overloaded or down.")
    except requests.exceptions.ConnectionError:
        st.error(f"🌐 Cannot connect to API at {API_URL}. Is the backend server running?")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")