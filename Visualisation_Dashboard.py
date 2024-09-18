import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

# Load your dataset from the pickle file
df = pd.read_pickle('data.pkl')

# Define the columns
numerical_columns = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
                     'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 
                     'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 
                     'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 
                     'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
                     'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
                     'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 
                     'Unemployment rate', 'Inflation rate', 'GDP']

categorical_columns = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance',
                       'Previous qualification', 'Nationality', 'Mother_qualification','Father_qualification', 
                       'Mother_occupation', 'Father_occupation', 'Displaced', 'Educational special needs', 
                       'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Target']

# Streamlit Sidebar
st.sidebar.title("Plotting Options")
plot_category = st.sidebar.selectbox("Select Plot Category", ["Univariate", "Bivariate", "Multivariate"])

# Univariate Plots
if plot_category == "Univariate":
    st.title("Univariate Plots")
    univariate_plot = st.sidebar.selectbox("Select Univariate Plot", ["Histogram (Numerical)", "Box Plot (Numerical)", "Count Plot (Categorical)"])

    # Histogram for numerical columns
    if univariate_plot == "Histogram (Numerical)":
        selected_col = st.sidebar.selectbox("Select Numerical Column", numerical_columns)
        st.subheader(f"Histogram for {selected_col}")
        
        fig = go.Figure(go.Histogram(
            x=df[selected_col], 
            nbinsx=15, 
            marker=dict(
                color='skyblue', 
                line=dict(color='black', width=1)
            )
        ))
        
        fig.update_layout(title=f"Histogram of {selected_col}")
        st.plotly_chart(fig)

    # Box plot for numerical columns
    elif univariate_plot == "Box Plot (Numerical)":
        selected_col = st.sidebar.selectbox("Select Numerical Column", numerical_columns)
        st.subheader(f"Box Plot for {selected_col}")
        
        fig = go.Figure(go.Box(
            y=df[selected_col],
            marker=dict(
                color='skyblue',
                line=dict(color='black', width=1)
            ),
            boxmean='sd'  # Optionally, show the mean and standard deviation
        ))
        
        fig.update_layout(title=f"Box Plot of {selected_col}")
        st.plotly_chart(fig)

    # Count plot for categorical columns
    elif univariate_plot == "Count Plot (Categorical)":
        selected_col = st.sidebar.selectbox("Select Categorical Column", categorical_columns)
        st.subheader(f"Count Plot for {selected_col}")
        
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=selected_col, palette="Set2")
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {selected_col}')
        st.pyplot(fig)

# Bivariate Plots
elif plot_category == "Bivariate":
    st.title("Bivariate Plots")
    bivariate_plot = st.sidebar.selectbox("Select Bivariate Plot", ["Scatter Plot (Numerical vs Numerical)"])

    if bivariate_plot == "Scatter Plot (Numerical vs Numerical)":
        x_col = st.sidebar.selectbox("Select X-axis Column", numerical_columns)
        y_col = st.sidebar.selectbox("Select Y-axis Column", numerical_columns)
        st.subheader(f"Scatter Plot: {x_col} vs {y_col}")
        
        fig = go.Figure(go.Scatter(
            x=df[x_col], y=df[y_col],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.7)
        ))
        
        fig.update_layout(title=f"Scatter Plot of {x_col} vs {y_col}")
        st.plotly_chart(fig)

# Multivariate Plots
elif plot_category == "Multivariate":
    st.title("Multivariate Plots")
    multivariate_plot = st.sidebar.selectbox("Select Multivariate Plot", ["Correlation Heatmap", "Scatter Plot with Target"])

    # Correlation Heatmap
    if multivariate_plot == "Correlation Heatmap":
        st.subheader("Correlation Heatmap of Numerical Features")
        numeric_df = df[numerical_columns]
        corr_matrix = numeric_df.corr()

        heatmap = go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation'),
            showscale=True,
            text=np.round(corr_matrix.values, 2),
            hoverinfo='text',
            texttemplate="%{text}"
        )

        fig = go.Figure(data=[heatmap])

        fig.update_layout(
            title='Correlation Matrix Heatmap',
            title_x=0.5,
            width=800, height=800
        )
        st.plotly_chart(fig)

    # Scatter plot with target as color encoding
    elif multivariate_plot == "Scatter Plot with Target":
        num_pairs = [('Previous qualification (grade)', 'Admission grade'),
                     ('Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)')]

        selected_pair = st.sidebar.selectbox("Select Variable Pair", [f"{x[0]} vs {x[1]}" for x in num_pairs])
        selected_pair = num_pairs[[f"{x[0]} vs {x[1]}" for x in num_pairs].index(selected_pair)]
        
        st.subheader(f"Scatter Plot: {selected_pair[0]} vs {selected_pair[1]}")
        
        color_palette = {'Dropout': 'red', 'Enrolled': 'blue', 'Graduate': 'green'}
        fig = go.Figure(go.Scatter(
            x=df[selected_pair[0]],
            y=df[selected_pair[1]],
            mode='markers',
            marker=dict(
                color=df['Target'].map(color_palette),
                size=5,
                opacity=0.7
            ),
            text=df['Target']
        ))

        fig.update_layout(title=f"Scatter Plot of {selected_pair[0]} vs {selected_pair[1]}")
        st.plotly_chart(fig)
