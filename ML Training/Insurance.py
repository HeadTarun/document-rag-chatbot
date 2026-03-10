import streamlit as st
import numpy as np
import pickle

# load model
model = pickle.load(open("ikr.pkl","rb"))

st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="💊",
    layout="wide"
)

st.title("💊 AI Health Insurance Cost Predictor")
st.markdown("Estimate medical insurance charges using AI risk analysis")

st.divider()

# Sidebar info
st.sidebar.header("Insurance Risk Factors")
st.sidebar.write("""
The AI estimates insurance charges based on:

• Age  
• BMI (Body Mass Index)  
• Smoking habits  
• Family dependents  
• Region  
""")

# Layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("👤 Personal Information")

    age = st.slider("Age", 18, 65, 30)

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    bmi = st.slider(
        "BMI (Body Mass Index)",
        10.0, 50.0, 25.0
    )

with col2:

    st.subheader("🏥 Health & Family")

    children = st.slider(
        "Number of Children",
        0, 5, 0
    )

    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

st.divider()

# Encoding values
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# prediction
if st.button("Estimate Insurance Cost 🧠"):

    features = np.array([[age,
                          sex_val,
                          bmi,
                          children,
                          smoker_val,
                          region_northwest,
                          region_southeast,
                          region_southwest]])

    prediction = model.predict(features)[0]

    st.success("### Estimated Insurance Charges")

    st.metric(
        label="Predicted Medical Cost",
        value=f"${prediction:,.2f}"
    )

    st.balloons()

st.markdown("---")
st.caption("AI Health Risk Analyzer • Built with Streamlit")