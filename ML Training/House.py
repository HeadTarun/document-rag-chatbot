import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# load model
model = pickle.load(open("model.pkl", "rb"))

# recreate polynomial transformer
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit([[0]*8])   # define 8 input features

# Load model


# Page settings
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Title
st.title("🏡 AI House Price Prediction System")
st.markdown("Predict property prices using Machine Learning")

st.divider()

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This AI model predicts **house prices** based on:

• Location coordinates  
• Population density  
• Room statistics  
• Median income  

Built with **Python, Scikit-Learn, and Streamlit**
""")

# Layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("📍 Location Information")

    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)

    housing_median_age = st.slider(
        "Housing Median Age",
        1, 100, 20
    )

    median_income = st.slider(
        "Median Income",
        0.0, 15.0, 3.0
    )


with col2:

    st.subheader("🏠 Property Details")

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=2000
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=400
    )

    population = st.number_input(
        "Population",
        min_value=1,
        value=1000
    )

    households = st.number_input(
        "Households",
        min_value=1,
        value=300
    )


st.divider()

# Prediction button
if st.button("Predict Price 💰"):

    features = np.array([[longitude,
                          latitude,
                          housing_median_age,
                          total_rooms,
                          total_bedrooms,
                          population,
                          households,
                          median_income]])

    # convert 8 → 44 features
    poly_features = poly.transform(features)

    prediction = model.predict(poly_features)

    st.success(f"Estimated Price:${prediction[0].astype(float)}")
   

    st.balloons()

st.markdown("---")
st.caption("Powered by Machine Learning • Built with Streamlit")