import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Building Energy Efficiency Predictor",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Building Energy Efficiency Prediction")
st.write("Predict heating & cooling load based on structural parameters.")

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.sidebar.header("📋 Input Building Parameters")

relative_compactness = st.sidebar.slider("Relative Compactness", 0.60, 1.00, 0.75, 0.01)
surface_area = st.sidebar.number_input("Surface Area (m²)", 500.0, 900.0, 670.0, 5.0)
wall_area = st.sidebar.number_input("Wall Area (m²)", 200.0, 500.0, 300.0, 5.0)
roof_area = st.sidebar.number_input("Roof Area (m²)", 100.0, 300.0, 150.0, 5.0)
overall_height = st.sidebar.selectbox("Overall Height (m)", [3.5, 7.0], index=1)
orientation = st.sidebar.selectbox("Orientation", [2, 3, 4, 5], format_func=lambda x: {2: "North", 3: "East", 4: "South", 5: "West"}[x])
glazing_area = st.sidebar.select_slider("Glazing Area Ratio", [0.0, 0.10, 0.25, 0.40], value=0.25)
glazing_area_distribution = st.sidebar.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

input_data = pd.DataFrame([[
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    overall_height,
    orientation,
    glazing_area,
    glazing_area_distribution
]], columns=[
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution"
])

st.subheader("📊 Input Features")
st.dataframe(input_data, use_container_width=True)

if st.button("🚀 Predict Energy Load", type="primary", use_container_width=True):
    if model is not None:
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Energy Load: **{prediction:.2f} kWh/m²**")
    else:
        st.error("Model could not be loaded.")
