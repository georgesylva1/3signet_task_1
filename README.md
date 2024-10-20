# Student Dropout Prediction App

An interactive web application that predicts student dropout likelihood based on academic and personal features. The app uses a trained Random Forest model to provide insights into whether a student is likely to drop out, stay enrolled, or graduate.

### 📋 Table of Contents
- Introduction
- Features
- Installation
- Usage
Model Overview
Technologies Used
Project Structure
Contributing
License
🚀 Introduction
Predicting student dropout can help educational institutions identify students at risk and provide timely interventions. This app takes in academic data and personal details to predict the likelihood of dropout. It provides an easy-to-use interface built with Streamlit for interactive predictions.

✨ Features
Interactive Interface: Users can input features such as course enrollment, grades, and curricular performance via dropdowns and sliders.
Real-time Predictions: Predict whether a student is likely to drop out or stay enrolled, with instant feedback.
User-friendly Outputs: Displays clear results, e.g., "High likelihood of Dropout" or "Not Likely to Dropout."
Input Features:
Categorical: Course, Mother's Occupation, Tuition Fees Up-to-Date
Numerical: Previous Qualification Grade, Admission Grade, Age at Enrollment, Performance in Curricular Units
🛠 Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/student-dropout-prediction-app.git
cd student-dropout-prediction-app
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the Application
bash
Copy code
streamlit run app.py
Once the server starts, the app will be available in your browser at http://localhost:8501.

📊 Usage
Open the app in your browser.
Input the student's academic and personal information using the dropdowns and sliders.
Press the "Predict" button to generate a prediction.
The prediction will indicate whether the student has a High likelihood of Dropout or is Not Likely to Dropout.
Example Prediction:
