# Student Dropout Prediction App

An interactive web application that predicts student dropout likelihood based on academic and personal features. The app uses a trained Random Forest model to provide insights into whether a student is likely to drop out, stay enrolled, or graduate.

### 🚀 Introduction
Predicting student dropout can help educational institutions identify students at risk and provide timely interventions. This app takes in academic data and personal details to predict the likelihood of dropout. It also provides an easy-to-use interface built with Streamlit for interactive predictions.

### ✨ Features
**Interactive Interface**: Users can input features such as course enrollment, grades, and curricular performance via dropdowns and sliders.

**Real-time Predictions**: Predict whether a student is likely to drop out or stay enrolled, with instant feedback.

**User-friendly Outputs**: Displays clear results, e.g., "High likelihood of Dropout" or "Not Likely to Dropout."

### Input Features:

**Categorical**: Course, Mother's Occupation, Tuition Fees Up-to-Date

**Numerical**: Previous qualification (grade), Admission grade, Age at enrollment, Curricular units 1st sem (enrolled), Curricular units 1st sem (evaluations),
Curricular units 1st sem (approved), Curricular units 1st sem (grade), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (evaluations),
Curricular units 2nd sem (approved), Curricular units 2nd sem (grade), GDP.

📊 Usage
Open the app in your browser. url is (https://3signettask1-version3.streamlit.app/)
Input the student's academic and personal information using the dropdowns and sliders.

Press the "Predict" button to generate a prediction.
The prediction will indicate whether the student has a High likelihood of Dropout or is Not Likely to Dropout.


### Data Analysis and Model Evaluation
Before deploying the predictive model in a Streamlit app, a comprehensive data analysis and model evaluation process was carried out to ensure accuracy and performance. 

To see the relationships between the various features before I had to drop the less important ones check the url: (https://3signettask1-visuals2.streamlit.app/)

**Data Preprocessing**: The dataset was cleaned, with missing values imputed and categorical features encoded. Numerical features were scaled to ensure consistency.
Feature Engineering: Key features were selected for model training, including both categorical and numerical variables (e.g., 'Course', 'Admission grade', 'Age at enrollment', 'Curricular units' etc).

**Model Training**: Several models were trained, including Random Forest and XGBoost. These were fine-tuned using RandomizedSearchCV for optimal performance.
Cross-Validation and Evaluation: The models were evaluated using cross-validation and metrics such as accuracy, precision, recall, and F1-score to ensure generalization. 
The final model achieved strong performance on the test set.

The best-performing model was saved and integrated into the Streamlit app, allowing users to make real-time predictions based on new data inputs.


