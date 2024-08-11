import os
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyreadstat
import pickle

# Set up the folder path where models are stored
folder_path = '/Users/foadk/Downloads/Pyhton_v2'

# Load models
def load_model(model_name):
    model_path = os.path.join(folder_path, model_name)
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model1 = load_model('model1.pkl')
model2 = load_model('model2.pkl')
model4 = load_model('model4.pkl')

# Function to process inputs for model1
def process_inputs_model1(admission_source, kps, nses):
    return pd.DataFrame({
        'const': 1,  # Add an intercept if your model expects it
        'ERAdmit': [1 if admission_source == "Non-elective admission" else 0],
        'KPSscore': [kps],
        'NSESindex': [nses],
    })

# Function to process inputs for model2
def process_inputs_model2(insurance, admission_source, mfi, nses):
    return pd.DataFrame({
        'const': 1,  # Add an intercept if your model expects it
        'insurancebinaryfinal': [1 if insurance == "Medicare/Medicaid/Uninsured/other" else 0],
        'ERAdmit': [1 if admission_source == "Non-elective admission" else 0],
        'MFI': [mfi],
        'NSESindex': [nses],
    })

# Function to process inputs for model4
def process_inputs_model4(age, insurance, kps, nses, state_of_residence):
    return pd.DataFrame({
        'const': 1,  # Add an intercept if your model expects it
        'Age': [age],
        'insurancebinaryfinal': [1 if insurance == "Medicare/Medicaid/Uninsured/other" else 0],
        'KPSscore': [kps],
        'NSESindex': [nses],
        'MDandotherneighbors': [1 if state_of_residence == "Other states" else 0]
    })

# Streamlit UI and other code remains the same...



# Streamlit UI
st.title("Glioblastoma High-Value Care Outcomes Calculator")

st.sidebar.header("Input Parameters")
age = st.sidebar.slider("Age", min_value=18, max_value=90, value=72)
insurance = st.sidebar.selectbox("Please select type of insurance:", 
                                 options=["Private", "Medicare/Medicaid/Uninsured/other"])
state_of_residence = st.sidebar.selectbox("Please select state of residence:", 
                                          options=["MD and neighboring states", "Other states"])
admission_source = st.sidebar.selectbox("Please select admission source:", 
                                        options=["Elective admission", "Non-elective admission"])
mfi = st.sidebar.slider("mFI-5", min_value=0, max_value=5, value=5)
kps = st.sidebar.slider("KPS", min_value=20, max_value=100, value=80, step=10)
nses = st.sidebar.slider("NSES", min_value=25, max_value=90, value=65)

if st.sidebar.button("Submit"):
    # Process inputs and predict
    new_data1 = process_inputs_model1(admission_source, kps, nses)
    new_data2 = process_inputs_model2(insurance, admission_source, mfi, nses)
    new_data4 = process_inputs_model4(age, insurance, kps, nses, state_of_residence)
    
    pred1 = model1.predict(new_data1)[0] * 100
    pred2 = model2.predict(new_data2)[0] * 100
    pred4 = model4.predict(new_data4)[0] * 100
    
    st.subheader("Prediction Results")
    st.write(f"Probability of extended length of stay: {pred1:.2f}%")
    st.write(f"Probability of non-routine discharge disposition: {pred2:.2f}%")
    st.write(f"Probability of non-initiation of Stupp protocol: {pred4:.2f}%")
else:
    st.subheader("The calculator is ready for your input.")

# About Section
st.sidebar.header("About")
st.sidebar.write("""
This web application provides a calculator that utilizes neighborhood socioeconomic status (NSES) 
index scores to predict high-value care outcomes following glioblastoma resection.

The calculator is intended for use by healthcare providers and researchers and is not designed 
to be used as a diagnostic tool. Please consult a healthcare provider for diagnosis and treatment 
of medical conditions.
""")
