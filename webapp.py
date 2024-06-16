
"""
Created on Thu Jun 13 10:31:26 2024

@author: user
"""
import pickle as pk
import pandas as pd
import numpy as np
import streamlit as st

# File paths
model_path = "C:/Users/user/karani/diabetes_Model.sav"
data_path = "C:/Users/user/karani/diabetes.csv"

# Load the model
try:
    with open(model_path, "rb") as file:
        model = pk.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the data
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Data file not found at {data_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading the data: {e}")
    st.stop()

st.header("Diabetes Predictor")

# Input fields
Pregnancies = st.number_input("Enter Number of Pregnancies", min_value=0)
Glucose = st.number_input("Enter Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("Enter BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Enter Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Enter Age", min_value=0)

# Prepare the input data
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

if st.button("Predict"):
    try:
        output = model.predict(input_data)
        if output[0] == 0:
            stn = "The patient has no Diabetes"
        else:
            stn = "The patient has Diabetes"
        st.markdown(stn)
    except Exception as e:
        st.error(f"Error making prediction: {e}")