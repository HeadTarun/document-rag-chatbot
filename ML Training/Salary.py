import streamlit as st
import numpy as np
import pickle

# load model
model = pickle.load(open("salary_model.pkl","rb"))

st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Career Salary Predictor")
st.markdown("Estimate your potential salary using Machine Learning")

st.divider()

# Sidebar
st.sidebar.header("About this Model")
st.sidebar.write("""
This AI model predicts **salary** based on:

• Age  
• Education Level  
• Work Experience  
• Gender
""")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")

    age = st.slider(
        "Age",
        18, 65, 25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    st.subheader("🎓 Career Information")

    education = st.slider(
        "Education Level",
        1, 5, 3,
        help="1=High School, 5=PhD"
    )

    experience = st.slider(
        "Years of Experience",
        0, 40, 2
    )

st.divider()

# convert gender
gender_male = 1 if gender == "Male" else 0

if st.button("Predict My Salary 🚀"):

    features = np.array([[age,
                          education,
                          experience,
                          gender_male]])

    prediction = model.predict(features)[0]

    st.success("### Estimated Salary")

    st.metric(
        label="Predicted Salary",
        value=f"${prediction:,.2f}"
    )

    st.balloons()

st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Salary Estimator")