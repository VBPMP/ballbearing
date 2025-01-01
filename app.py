import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load your CSV file
df = pd.read_csv(r"Vasant_Project.csv")

# Define target and selected columns
target_column = ['Bearing_No', 'MAX_Stress', 'MAX_Deformation']
selected_columns = ['Limitng_speed', 'Shaft_Diameter', 'Bearing_No', 'MAX_Stress', 'MAX_Deformation']

# Filter and drop NaN values
df_selected = df[selected_columns].dropna()

# Separate features and target
X = df_selected.drop(target_column, axis=1)
y = df_selected[target_column]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load the saved model
rf_model = joblib.load('ball_bearing_rf_model.pkl')

# Streamlit GUI
st.title("Ball Bearing Selection Tool")

# User input for features
limiting_speed = st.slider(
    "Select Limiting Speed",
    float(X['Limitng_speed'].min()),
    float(X['Limitng_speed'].max())
)
shaft_diameter = st.slider(
    "Select Shaft Diameter",
    float(X['Shaft_Diameter'].min()),
    float(X['Shaft_Diameter'].max())
)

# Make prediction
input_data = scaler.transform([[limiting_speed, shaft_diameter]])
prediction = rf_model.predict(input_data)

# Display prediction
st.write("Predicted Output:")
st.write(f"MAX_Stress: {prediction[0][1]:.2f}")
st.write(f"MAX_Deformation: {prediction[0][2]:.8f}")
st.write(f"Bearing_No: {int(prediction[0][0])}")
