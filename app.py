
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


model = joblib.load('employee_attrition_model.pkl')


scaler = StandardScaler()


st.title("Employee Attrition Prediction")


Age = st.number_input("Age", min_value=18, max_value=60, value=30)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
Gender = st.selectbox("Gender", ["Male", "Female"])
YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)


Gender = 0 if Gender == "Male" else 1


input_data = pd.DataFrame([[Age, MonthlyIncome, Gender, YearsAtCompany]], 
                          columns=['Age', 'MonthlyIncome', 'Gender', 'YearsAtCompany'])


input_data[['Age', 'MonthlyIncome']] = scaler.fit_transform(input_data[['Age', 'MonthlyIncome']])


if st.button("Predict Attrition"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("The employee is likely to leave.")
    else:
        st.write("The employee is likely to stay.")
