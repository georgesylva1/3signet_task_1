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
    st.title("Student Dropout Prediction App")
    st.write("Enter the features below and click 'Predict' to get a prediction.")
    
    # Dropdowns for categorical features
    marital_status = st.selectbox("Marital status", sorted(df['Marital status'].unique()))
    application_mode = st.selectbox("Application mode", sorted(df['Application mode'].unique()))
    application_order = st.selectbox("Application order", sorted(df['Application order'].unique()))
    course = st.selectbox("Course", sorted(df['Course'].unique()))
    daytime_evening_attendance = st.selectbox("Daytime/evening attendance", sorted(df['Daytime/evening attendance'].unique()))
    previous_qualification = st.selectbox("Previous qualification", sorted(df['Previous qualification'].unique()))
    nationality = st.selectbox("Nationality", sorted(df['Nationality'].unique()))
    mother_qualification = st.selectbox("Mother's qualification", sorted(df['Mother_qualification'].unique()))
    father_qualification = st.selectbox("Father's qualification", sorted(df['Father_qualification'].unique()))
    mother_occupation = st.selectbox("Mother's occupation", sorted(df['Mother_occupation'].unique()))
    father_occupation = st.selectbox("Father's occupation", sorted(df['Father_occupation'].unique()))
    displaced = st.selectbox("Displaced", sorted(df['Displaced'].unique()))
    educational_special_needs = st.selectbox("Educational special needs", sorted(df['Educational special needs'].unique()))
    debtor = st.selectbox("Debtor", sorted(df['Debtor'].unique()))
    tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", sorted(df['Tuition fees up to date'].unique()))
    gender = st.selectbox("Gender", sorted(df['Gender'].unique()))
    scholarship_holder = st.selectbox("Scholarship holder", sorted(df['Scholarship holder'].unique()))
    international = st.selectbox("International", sorted(df['International'].unique()))
    
    # Sliders for numerical features
    previous_qualification_grade = st.slider("Previous qualification (grade)", min_value=float(df['Previous qualification (grade)'].min()), max_value=float(df['Previous qualification (grade)'].max()), step=0.1)
    admission_grade = st.slider("Admission grade", min_value=float(df['Admission grade'].min()), max_value=float(df['Admission grade'].max()), step=0.1)
    age_at_enrollment = st.slider("Age at enrollment", min_value=int(df['Age at enrollment'].min()), max_value=int(df['Age at enrollment'].max()), step=1)
    curricular_units_1st_sem_credited = st.slider("Curricular units 1st sem (credited)", min_value=int(df['Curricular units 1st sem (credited)'].min()), max_value=int(df['Curricular units 1st sem (credited)'].max()), step=1)
    curricular_units_1st_sem_enrolled = st.slider("Curricular units 1st sem (enrolled)", min_value=int(df['Curricular units 1st sem (enrolled)'].min()), max_value=int(df['Curricular units 1st sem (enrolled)'].max()), step=1)
    curricular_units_1st_sem_evaluations = st.slider("Curricular units 1st sem (evaluations)", min_value=int(df['Curricular units 1st sem (evaluations)'].min()), max_value=int(df['Curricular units 1st sem (evaluations)'].max()), step=1)
    curricular_units_1st_sem_approved = st.slider("Curricular units 1st sem (approved)", min_value=int(df['Curricular units 1st sem (approved)'].min()), max_value=int(df['Curricular units 1st sem (approved)'].max()), step=1)
    curricular_units_1st_sem_grade = st.slider("Curricular units 1st sem (grade)", min_value=float(df['Curricular units 1st sem (grade)'].min()), max_value=float(df['Curricular units 1st sem (grade)'].max()), step=0.1)
    curricular_units_1st_sem_without_evaluations = st.slider("Curricular units 1st sem (without evaluations)", min_value=int(df['Curricular units 1st sem (without evaluations)'].min()), max_value=int(df['Curricular units 1st sem (without evaluations)'].max()), step=1)
    curricular_units_2nd_sem_credited = st.slider("Curricular units 2nd sem (credited)", min_value=int(df['Curricular units 2nd sem (credited)'].min()), max_value=int(df['Curricular units 2nd sem (credited)'].max()), step=1)
    curricular_units_2nd_sem_enrolled = st.slider("Curricular units 2nd sem (enrolled)", min_value=int(df['Curricular units 2nd sem (enrolled)'].min()), max_value=int(df['Curricular units 2nd sem (enrolled)'].max()), step=1)
    curricular_units_2nd_sem_evaluations = st.slider("Curricular units 2nd sem (evaluations)", min_value=int(df['Curricular units 2nd sem (evaluations)'].min()), max_value=int(df['Curricular units 2nd sem (evaluations)'].max()), step=1)
    curricular_units_2nd_sem_approved = st.slider("Curricular units 2nd sem (approved)", min_value=int(df['Curricular units 2nd sem (approved)'].min()), max_value=int(df['Curricular units 2nd sem (approved)'].max()), step=1)
    curricular_units_2nd_sem_grade = st.slider("Curricular units 2nd sem (grade)", min_value=float(df['Curricular units 2nd sem (grade)'].min()), max_value=float(df['Curricular units 2nd sem (grade)'].max()), step=0.1)
    curricular_units_2nd_sem_without_evaluations = st.slider("Curricular units 2nd sem (without evaluations)", min_value=int(df['Curricular units 2nd sem (without evaluations)'].min()), max_value=int(df['Curricular units 2nd sem (without evaluations)'].max()), step=1)
    unemployment_rate = st.slider("Unemployment rate", min_value=float(df['Unemployment rate'].min()), max_value=float(df['Unemployment rate'].max()), step=0.1)
    inflation_rate = st.slider("Inflation rate", min_value=float(df['Inflation rate'].min()), max_value=float(df['Inflation rate'].max()), step=0.1)
    gdp = st.slider("GDP", min_value=float(df['GDP'].min()), max_value=float(df['GDP'].max()), step=0.1)

    # Create a button to trigger prediction
    if st.button("Predict"):
        # Create a DataFrame with user input for all features
        new_data = pd.DataFrame([[marital_status, application_mode, application_order, course,
                                daytime_evening_attendance, previous_qualification,
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
                                curricular_units_2nd_sem_without_evaluations, unemployment_rate,
                                inflation_rate, gdp]], 
                               columns=['Marital status', 'Application mode', 'Application order', 'Course',
                                        'Daytime/evening attendance', 'Previous qualification',
                                        'Previous qualification (grade)', 'Nationality',
                                        'Mother_qualification', 'Father_qualification',
                                        'Mother_occupation', 'Father_occupation', 'Admission grade',
                                        'Displaced', 'Educational special needs', 'Debtor',
                                        'Tuition fees up to date', 'Gender', 'Scholarship holder',
                                        'Age at enrollment', 'International',
                                        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 
                                        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
                                        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
                                        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
                                        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
                                        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
                                        'Unemployment rate', 'Inflation rate', 'GDP'])
        
              
        # Make a prediction
        prediction = rf_pipeline.predict(new_data_preprocessed)[0]
        
        # Get the corresponding label for the predicted class
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        # Display the prediction and its corresponding label
        st.write("Prediction (Class):", prediction)
        st.write("Prediction (Label):", prediction_label)

if __name__ == "__main__":
    main()
