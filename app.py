import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time


@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("medical_cost_pred_big_data_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
model, scalar = load_model_and_scaler()

@st.cache_data
def data_input2(age,sex,bmi,children,smoker,region):
    Age=age
    Sex=sex.lower()
    if Sex=="male":
        Sex=1
    else:
        Sex=0
    Bmi=bmi
    Children=children
    Smoker=smoker
    if Smoker=="yes":
        Smoker=1
    else:
        Smoker=0
    Region=region
    if region=="northeast":
        Region=[0,0,0]
    elif region=="northwest":
        Region=[1,0,0]
    elif region=="southeast":
        Region=[0,1,0]
    else:
        Region=[0,0,1]
    return Age,Sex,Bmi,Children,Smoker,Region


st.title("Medical Insurance Cost Prediction")
with st.form("my_form"):
    age = st.slider("Age",18,65,25)
    sex = st.selectbox("Sex",("male","female"))
    bmi = st.number_input("BMI",min_value=0.0,max_value=40.0,value=25.0)
    children = st.selectbox("Children",("0","1","2","3","4","5"),index=0)
    smoker = st.selectbox("Smoker",("yes","no"))
    region = st.selectbox("Region",("northeast","northwest","southeast","southwest"))
    submit = st.form_submit_button("Predict")
    if submit:
        Age,Sex,Bmi,Children,Smoker,Region=data_input2(age,sex,bmi,int(children),smoker,region)

        input_data = np.array([Age , Sex , Bmi, Children , Smoker ] + Region)
        input_data = np.array(input_data, dtype=np.float32)

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        
        for percent_complete in range(0, 101, 25):
            time.sleep(0.04) 
            progress_bar.progress(percent_complete)
            status_text.text(f"Processing... {percent_complete}%")
        status_text.text("Processing... Done!")
            
        input_data = scalar.transform([input_data])
        input_data_reshaped = input_data.reshape(1,-1)
        prediction = model.predict(input_data_reshaped)

        st.success(f"The predicted medical insurance cost is : {prediction[0]:.2f} ")