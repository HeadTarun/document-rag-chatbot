import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Grade Prediction System")
st.write("Predict student performance using ML")

st.divider()

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
"""
This ML model predicts student **grades** based on:

• Self study hours  
• Attendance  
• Class participation  
• Total score
"""
)

# Input Section
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    student_id = st.number_input("Student ID", min_value=1, step=1)
    study_hours = st.slider("Weekly Self Study Hours", 0, 40, 5)

with col2:
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    participation = st.slider("Class Participation", 0, 10, 5)
    

total_score = st.slider("Total Score", 0, 100, 50)

st.divider()
def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
     return 'F'

# Prediction
if st.button("Predict Grade 🚀"):

    features = np.array([[student_id ,study_hours,
                          attendance,
                          participation,
                          total_score]])

    prediction = model.predict(features)[0]
    grad = get_grade(prediction)
    st.success(f"Predicted Grade: **{prediction}** | Grade : {grad}")

    st.balloons()

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Machine Learning")