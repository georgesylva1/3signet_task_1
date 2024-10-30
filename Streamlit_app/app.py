import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Load the saved LabelEncoder and Random Forest pipeline
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

with open('rf_pipeline.pkl', 'rb') as f:
    rf_pipeline = joblib.load(f)

# Load the DataFrame from the pickle file
df = pd.read_pickle('data.pkl')

# Identify categorical columns
categorical_cols = df.select_dtypes(include='category').columns

# Convert categorical columns to numerical using LabelEncoder
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Identify numerical columns (excluding converted categorical columns)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(categorical_cols)

# Create a Streamlit app
def main():
    st.title("Random Forest Model Prediction")
    st.write("Enter the features below and click 'Predict' to get a prediction.")
    
    # Add input fields for all features
    marital_status = st.selectbox("Marital status", df['Marital status'].unique())
    application_mode = st.selectbox("Application mode", df['Application mode'].unique())
    application_order = st.number_input("Application order")
    course = st.selectbox("Course", df['Course'].unique())
    daytime_evening_attendance = st.selectbox("Daytime/evening attendance", df['Daytime/evening attendance'].unique())
    previous_qualification = st.selectbox("Previous qualification", df['Previous qualification'].unique())
    previous_qualification_grade = st.number_input("Previous qualification (grade)")
    nationality = st.selectbox("Nationality", df['Nationality'].unique())
    mother_qualification = st.selectbox("Mother's qualification", df['Mother_qualification'].unique())
    father_qualification = st.selectbox("Father's qualification", df['Father_qualification'].unique())
    mother_occupation = st.selectbox("Mother's occupation", df['Mother_occupation'].unique())
    father_occupation = st.selectbox("Father's occupation", df['Father_occupation'].unique())
    admission_grade = st.number_input("Admission grade")
    displaced = st.selectbox("Displaced", df['Displaced'].unique())
    educational_special_needs = st.selectbox("Educational special needs", df['Educational special needs'].unique())
    debtor = st.selectbox("Debtor", df['Debtor'].unique())
    tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", df['Tuition fees up to date'].unique())
    gender = st.selectbox("Gender", df['Gender'].unique())
    scholarship_holder = st.selectbox("Scholarship holder", df['Scholarship holder'].unique())
    age_at_enrollment = st.number_input("Age at enrollment")
    international = st.selectbox("International", df['International'].unique())
    curricular_units_1st_sem_credited = st.number_input("Curricular units 1st sem (credited)")
    curricular_units_1st_sem_enrolled = st.number_input("Curricular units 1st sem (enrolled)")
    curricular_units_1st_sem_evaluations = st.number_input("Curricular units 1st sem (evaluations)")
    curricular_units_1st_sem_approved = st.number_input("Curricular units 1st sem (approved)")
    curricular_units_1st_sem_grade = st.number_input("Curricular units 1st sem (grade)")
    curricular_units_1st_sem_without_evaluations = st.number_input("Curricular units 1st sem (without evaluations)")
    curricular_units_2nd_sem_credited = st.number_input("Curricular units 2nd sem (credited)")
    curricular_units_2nd_sem_enrolled = st.number_input("Curricular units 2nd sem (enrolled)")
    curricular_units_2nd_sem_evaluations = st.number_input("Curricular units 2nd sem (evaluations)")
    curricular_units_2nd_sem_approved = st.number_input("Curricular units 2nd sem (approved)")
    curricular_units_2nd_sem_grade = st.number_input("Curricular units 2nd sem (grade)")
    curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular units 2nd sem (without evaluations)")
    unemployment_rate = st.number_input("Unemployment rate")
    inflation_rate = st.number_input("Inflation rate")
    gdp = st.number_input("GDP")


    # Create a button to trigger prediction
    if st.button("Predict"):
        # Create a DataFrame with user input for all features
        new_data = pd.DataFrame([[marital_status, application_mode, application_order,
                                course, daytime_evening_attendance, previous_qualification,
                                previous_qualification_grade, nationality, mother_qualification,
                                father_qualification, mother_occupation, father_occupation,
                                admission_grade, displaced, educational_special_needs, debtor,
                                tuition_fees_up_to_date, gender, scholarship_holder,
                                age_at_enrollment, international, curricular_units_1st_sem_credited,
                                curricular_units_1st_sem_enrolled, curricular_units_1st_sem_evaluations,
                                curricular_units_1st_sem_approved, curricular_units_1st_sem_grade,
                                curricular_units_1st_sem_without_evaluations, curricular_units_2nd_sem_credited,
                                curricular_units_2nd_sem_enrolled, curricular_units_2nd_sem_evaluations,
                                curricular_units_2nd_sem_approved, curricular_units_2nd_sem_grade,
                                curricular_units_2nd_sem_without_evaluations, unemployment_rate, inflation_rate, gdp]], columns=df.columns)
           # Ensure the new_data is transformed like training data (e.g., encoding, scaling)
        # Make a prediction
        prediction = rf_pipeline.predict(new_data)[0]

        # Display the prediction
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
