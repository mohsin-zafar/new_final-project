"""
Manufacturing Equipment Output Prediction - Streamlit Frontend
===============================================================
This is the frontend application for the Manufacturing Output Prediction system.
It provides a user-friendly interface for making predictions.
"""

import streamlit as st
import requests
import json
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Manufacturing Output Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# BACKEND API URL
# ============================================
# Use environment variable for production, fallback to Render backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "https://manufacturing-prediction-backend.onrender.com")

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1565C0;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: #666;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🏭 Manufacturing Output Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict Parts Per Hour using Machine Learning</p>', unsafe_allow_html=True)

# Divider
st.markdown("---")

# ============================================
# SIDEBAR - API STATUS & INFO
# ============================================
with st.sidebar:
    st.header("📊 System Info")
    
    # Check API health
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Status: Online")
        else:
            st.warning("⚠️ API Status: Issues detected")
    except:
        st.error("❌ API Status: Offline")
        st.info("Make sure the backend is running on " + BACKEND_URL)
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.markdown("""
    This application predicts **Parts Per Hour** 
    output based on manufacturing machine parameters.
    
    **Model:** Linear Regression
    
    **Features Used:**
    - Machine Parameters
    - Environmental Factors
    - Operator Information
    - Process Metrics
    """)
    
    st.markdown("---")
    
    st.header("🔧 Backend URL")
    backend_url_input = st.text_input("API URL", value=BACKEND_URL)
    if backend_url_input != BACKEND_URL:
        BACKEND_URL = backend_url_input
        st.experimental_rerun()

# ============================================
# INPUT FORM
# ============================================
st.header("📝 Enter Manufacturing Parameters")

# Create columns for organized input
col1, col2, col3 = st.columns(3)

# ============================================
# COLUMN 1 - PROCESS PARAMETERS
# ============================================
with col1:
    st.subheader("🔥 Process Parameters")
    
    injection_temperature = st.number_input(
        "Injection Temperature (°C)",
        min_value=150.0,
        max_value=300.0,
        value=220.0,
        step=1.0,
        help="Temperature during injection process"
    )
    
    injection_pressure = st.number_input(
        "Injection Pressure (bar)",
        min_value=50.0,
        max_value=200.0,
        value=130.0,
        step=1.0,
        help="Pressure during injection"
    )
    
    cycle_time = st.number_input(
        "Cycle Time (seconds)",
        min_value=10.0,
        max_value=60.0,
        value=30.0,
        step=0.5,
        help="Total cycle time"
    )
    
    cooling_time = st.number_input(
        "Cooling Time (seconds)",
        min_value=3.0,
        max_value=30.0,
        value=12.0,
        step=0.5,
        help="Cooling phase duration"
    )
    
    material_viscosity = st.number_input(
        "Material Viscosity",
        min_value=50.0,
        max_value=600.0,
        value=300.0,
        step=10.0,
        help="Viscosity of material used"
    )
    
    temperature_pressure_ratio = st.number_input(
        "Temperature/Pressure Ratio",
        min_value=0.5,
        max_value=5.0,
        value=1.7,
        step=0.1,
        help="Ratio of temperature to pressure"
    )

# ============================================
# COLUMN 2 - MACHINE & ENVIRONMENT
# ============================================
with col2:
    st.subheader("⚙️ Machine & Environment")
    
    ambient_temperature = st.number_input(
        "Ambient Temperature (°C)",
        min_value=10.0,
        max_value=45.0,
        value=25.0,
        step=0.5,
        help="Surrounding temperature"
    )
    
    machine_age = st.number_input(
        "Machine Age (years)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Age of the machine"
    )
    
    maintenance_hours = st.number_input(
        "Hours Since Maintenance",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        help="Hours since last maintenance"
    )
    
    machine_type = st.selectbox(
        "Machine Type",
        options=["Type_A", "Type_B", "Type_C"],
        index=0,
        help="Type of manufacturing machine"
    )
    
    material_grade = st.selectbox(
        "Material Grade",
        options=["Economy", "Standard", "Premium"],
        index=1,
        help="Grade of material used"
    )
    
    total_cycle_time = st.number_input(
        "Total Cycle Time (seconds)",
        min_value=15.0,
        max_value=100.0,
        value=42.0,
        step=0.5,
        help="Total time for complete cycle"
    )

# ============================================
# COLUMN 3 - OPERATIONAL FACTORS
# ============================================
with col3:
    st.subheader("👷 Operational Factors")
    
    operator_experience = st.number_input(
        "Operator Experience (years)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Operator experience level"
    )
    
    shift = st.selectbox(
        "Work Shift",
        options=["Day", "Evening", "Night"],
        index=0,
        help="Current work shift"
    )
    
    day_of_week = st.selectbox(
        "Day of Week",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=0,
        help="Current day of the week"
    )
    
    efficiency_score = st.slider(
        "Efficiency Score",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Efficiency score (0-1)"
    )
    
    machine_utilization = st.slider(
        "Machine Utilization",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Machine utilization rate (0-1)"
    )

# ============================================
# PREDICTION BUTTON & RESULT
# ============================================
st.markdown("---")

# Center the predict button
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    predict_button = st.button("🔮 Predict Parts Per Hour", use_container_width=True)

# Make prediction when button is clicked
if predict_button:
    # Prepare input data
    input_data = {
        "Injection_Temperature": injection_temperature,
        "Injection_Pressure": injection_pressure,
        "Cycle_Time": cycle_time,
        "Cooling_Time": cooling_time,
        "Material_Viscosity": material_viscosity,
        "Ambient_Temperature": ambient_temperature,
        "Machine_Age": machine_age,
        "Operator_Experience": operator_experience,
        "Maintenance_Hours": maintenance_hours,
        "Shift": shift,
        "Machine_Type": machine_type,
        "Material_Grade": material_grade,
        "Day_of_Week": day_of_week,
        "Temperature_Pressure_Ratio": temperature_pressure_ratio,
        "Total_Cycle_Time": total_cycle_time,
        "Efficiency_Score": efficiency_score,
        "Machine_Utilization": machine_utilization
    }
    
    # Show spinner while making prediction
    with st.spinner("Making prediction..."):
        try:
            # Make API request
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=input_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result["success"]:
                    # Display prediction result
                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Result")
                    
                    # Create prediction display box
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <p class="prediction-label">Predicted Parts Per Hour</p>
                            <p class="prediction-value">{result['predicted_parts_per_hour']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Success message
                    st.success("✅ Prediction completed successfully!")
                    
                    # Show input summary
                    with st.expander("📋 View Input Summary"):
                        st.json(input_data)
                else:
                    st.error(f"❌ Prediction failed: {result['message']}")
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the backend API. Make sure it's running.")
            st.info(f"Backend URL: {BACKEND_URL}")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Manufacturing Equipment Output Prediction System</p>
    <p>Built with ❤️ using Streamlit & FastAPI</p>
</div>
""", unsafe_allow_html=True)
