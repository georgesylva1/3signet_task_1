import joblib
import pandas as pd
import streamlit as st

# Load the saved LabelEncoder and Random Forest pipeline
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

with open('rf_pipeline_model.pkl', 'rb') as f:
    rf_random_search2 = joblib.load(f)

# Load the DataFrame from the pickle file
df = pd.read_pickle('data.pkl')


# Create a Streamlit app
def main():
    st.title("Student Dropout Prediction App")
    st.write("Enter the features below and click 'Predict' to get a prediction.")

    # Dropdowns for categorical features that are in the selected features list
    course = st.selectbox("Course", sorted(df['Course'].unique()))
    mother_occupation = st.selectbox("Mother's occupation", sorted(df['Mother_occupation'].unique()))
    tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", sorted(df['Tuition fees up to date'].unique()))

    # Sliders for numerical features
    previous_qualification_grade = st.slider("Previous qualification (grade)",
                                             min_value=float(df['Previous qualification (grade)'].min()),
                                             max_value=float(df['Previous qualification (grade)'].max()), step=0.1)
    admission_grade = st.slider("Admission grade", min_value=float(df['Admission grade'].min()),
                                max_value=float(df['Admission grade'].max()), step=0.1)
    age_at_enrollment = st.slider("Age at enrollment", min_value=int(df['Age at enrollment'].min()),
                                  max_value=int(df['Age at enrollment'].max()), step=1)
    curricular_units_1st_sem_enrolled = st.slider("Curricular units 1st sem (enrolled)",
                                                  min_value=int(df['Curricular units 1st sem (enrolled)'].min()),
                                                  max_value=int(df['Curricular units 1st sem (enrolled)'].max()),
                                                  step=1)
    curricular_units_1st_sem_evaluations = st.slider("Curricular units 1st sem (evaluations)",
                                                     min_value=int(df['Curricular units 1st sem (evaluations)'].min()),
                                                     max_value=int(df['Curricular units 1st sem (evaluations)'].max()),
                                                     step=1)
    curricular_units_1st_sem_approved = st.slider("Curricular units 1st sem (approved)",
                                                  min_value=int(df['Curricular units 1st sem (approved)'].min()),
                                                  max_value=int(df['Curricular units 1st sem (approved)'].max()),
                                                  step=1)
    curricular_units_1st_sem_grade = st.slider("Curricular units 1st sem (grade)",
                                               min_value=float(df['Curricular units 1st sem (grade)'].min()),
                                               max_value=float(df['Curricular units 1st sem (grade)'].max()), step=0.1)
    curricular_units_2nd_sem_enrolled = st.slider("Curricular units 2nd sem (enrolled)",
                                                  min_value=int(df['Curricular units 2nd sem (enrolled)'].min()),
                                                  max_value=int(df['Curricular units 2nd sem (enrolled)'].max()),
                                                  step=1)
    curricular_units_2nd_sem_evaluations = st.slider("Curricular units 2nd sem (evaluations)",
                                                     min_value=int(df['Curricular units 2nd sem (evaluations)'].min()),
                                                     max_value=int(df['Curricular units 2nd sem (evaluations)'].max()),
                                                     step=1)
    curricular_units_2nd_sem_approved = st.slider("Curricular units 2nd sem (approved)",
                                                  min_value=int(df['Curricular units 2nd sem (approved)'].min()),
                                                  max_value=int(df['Curricular units 2nd sem (approved)'].max()),
                                                  step=1)
    curricular_units_2nd_sem_grade = st.slider("Curricular units 2nd sem (grade)",
                                               min_value=float(df['Curricular units 2nd sem (grade)'].min()),
                                               max_value=float(df['Curricular units 2nd sem (grade)'].max()), step=0.1)
    gdp = st.slider("GDP", min_value=float(df['GDP'].min()), max_value=float(df['GDP'].max()), step=0.1)

    # Create a button to trigger prediction
    if st.button("Predict"):
        # Create a DataFrame with user input for selected features
        new_data = pd.DataFrame([[course, previous_qualification_grade, mother_occupation, admission_grade,
                                  tuition_fees_up_to_date, age_at_enrollment, curricular_units_1st_sem_enrolled,
                                  curricular_units_1st_sem_evaluations, curricular_units_1st_sem_approved,
                                  curricular_units_1st_sem_grade, curricular_units_2nd_sem_enrolled,
                                  curricular_units_2nd_sem_evaluations, curricular_units_2nd_sem_approved,
                                  curricular_units_2nd_sem_grade, gdp]],
                                columns=['Course', 'Previous qualification (grade)', 'Mother_occupation',
                                         'Admission grade',
                                         'Tuition fees up to date', 'Age at enrollment',
                                         'Curricular units 1st sem (enrolled)',
                                         'Curricular units 1st sem (evaluations)',
                                         'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
                                         'Curricular units 2nd sem (enrolled)',
                                         'Curricular units 2nd sem (evaluations)',
                                         'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
                                         'GDP'])

        # Make a prediction
        prediction = rf_random_search2.predict(new_data)[0]

        # Get the corresponding label for the predicted class
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        # Check the predicted label and display a corresponding message
        if prediction_label == 'Dropout':
            st.write("Prediction: **High likelihood of Dropout**")
        else:
            st.write(f"Prediction: **Not Likely to Dropout**")


if __name__ == '__main__':
    main()
