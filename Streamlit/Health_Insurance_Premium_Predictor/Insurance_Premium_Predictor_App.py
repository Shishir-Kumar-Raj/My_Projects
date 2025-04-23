import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sidebar Navigation
navi = st.sidebar.radio("Navigation", ["About", "Predict"])

# Load and preprocess data
df = pd.read_csv('insurance.csv')
df.replace({'sex': {'male': 0, 'female': 1},
            'smoker': {'yes': 0, 'no': 1},
            'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

X = df.drop(columns='charges')
y = df['charges']

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

# About Section
if navi == "About":
    st.title("Health Insurance Premium Predictor")
    st.write("Predict your health insurance premium based on demographic and lifestyle information.")
    st.image('health_insurance.jpeg', width=550)

# Prediction Section
if navi == "Predict":
    st.title("Enter Your Details")

    age = st.number_input("Age", min_value=0, step=1)
    gender = st.radio("Gender", ("Male", "Female"))
    bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
    children = st.number_input("Number of Children", min_value=0, step=1)
    smoke = st.radio("Do you Smoke?", ("Yes", "No"))
    region = st.selectbox("Region", ("SouthEast", "SouthWest", "NorthEast", "NorthWest"))

    # Convert inputs to model-usable format
    gender_val = 0 if gender == "Male" else 1
    smoke_val = 0 if smoke == "Yes" else 1
    region_dict = {"SouthEast": 0, "SouthWest": 1, "NorthEast": 2, "NorthWest": 3}
    region_val = region_dict[region]

    if st.button("Predict"):
        input_data = [[age, gender_val, bmi, children, smoke_val, region_val]]
        prediction = model.predict(input_data)[0]
        st.subheader("Predicted Insurance Premium:")
        st.success(f"${prediction:,.2f}")
