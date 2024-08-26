import pandas as pd
import streamlit as st
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from io import BytesIO
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")  # Use the environment variable name here
if api_key is None:
    st.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key


def generate_description(query):
    """Generate a description or insights using OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=150  # Adjust the max tokens as per your requirement
        )
        description = response.choices[0].message['content'].strip()
        return description
    except Exception as e:
        st.error(f"Error generating description: {e}")
        return "Error generating description."



def load_data(uploaded_file):
    """Load the dataset from the uploaded CSV file."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        else:
            st.error("No file uploaded.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset by handling missing values, duplicates, and data types."""
    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        df = df.drop_duplicates()
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return df

def perform_eda(df, target_var):
    """Perform Exploratory Data Analysis and visualize the data."""
    st.write("Exploratory Data Analysis")

    try:
        # Correlation with target variable
        corr_with_target = df.corr()[target_var].sort_values(ascending=False)
        st.write("Correlation with Target Variable")
        st.write(corr_with_target)

        # Highly correlated features with target
        high_corr_features = corr_with_target[corr_with_target > 0.5].index

        if len(high_corr_features) > 0:
            fig, axes = plt.subplots(len(high_corr_features), 1, figsize=(10, 6 * len(high_corr_features)))
            fig.suptitle('Plots of Highly Correlated Features')
            for i, col in enumerate(high_corr_features):
                if i == 0:  # Skip the target variable itself
                    continue
                sns.histplot(df[col], ax=axes[i], color=sns.color_palette("Blues")[i % 10])
            st.pyplot(fig)
        else:
            st.write("No features highly correlated with the target variable.")

        # Subplots for all features
        fig, axes = plt.subplots(2, len(df.columns)//2, figsize=(20, 10))
        fig.suptitle('Subplots for All Features')
        for i, col in enumerate(df.columns):
            if df[col].dtype == 'object':
                sns.countplot(x=col, data=df, ax=axes[i//len(df.columns)//2, i % len(df.columns)//2], palette="Set3")
            else:
                sns.histplot(df[col], ax=axes[i//len(df.columns)//2, i % len(df.columns)//2], color=sns.color_palette("Blues")[i % 10])
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during EDA: {e}")

def feature_selection(df):
    """Perform feature selection by removing highly correlated features."""
    try:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        df = df.drop(columns=to_drop)
        st.write(f"Columns dropped due to high correlation: {to_drop}")
    except Exception as e:
        st.error(f"Feature selection error: {e}")
    return df

def train_model(df, x_vars, y_var, problem_type):
    """Train a model based on the problem type and evaluate its performance."""
    try:
        if y_var not in df.columns or not set(x_vars).issubset(df.columns):
            st.error("Invalid target or feature columns. Please check your input.")
            return

        X = df[x_vars]
        y = df[y_var]
        
        if X.select_dtypes(include=['object']).shape[1] > 0:
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if problem_type == "Regression":
            model_lr = LinearRegression()
            model_rfr = RandomForestRegressor()
            model_lr.fit(X_train, y_train)
            model_rfr.fit(X_train, y_train)
            predictions_lr = model_lr.predict(X_test)
            predictions_rfr = model_rfr.predict(X_test)
            st.write("Linear Regression Model Performance:")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions_lr)}")
            st.write(f"R-squared: {r2_score(y_test, predictions_lr)}")
            st.write("Random Forest Regressor Model Performance:")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions_rfr)}")
            st.write(f"R-squared: {r2_score(y_test, predictions_rfr)}")
        
        elif problem_type == "Classification":
            model_rf = RandomForestClassifier()
            model_lr = LogisticRegression(max_iter=1000)
            model_rf.fit(X_train, y_train)
            model_lr.fit(X_train, y_train)
            predictions_rf = model_rf.predict(X_test)
            predictions_lr = model_lr.predict(X_test)
            st.write("Random Forest Classifier Model Performance:")
            st.write(classification_report(y_test, predictions_rf))
            st.write(f"Accuracy: {accuracy_score(y_test, predictions_rf)}")
            st.write("Logistic Regression Model Performance:")
            st.write(classification_report(y_test, predictions_lr))
            st.write(f"Accuracy: {accuracy_score(y_test, predictions_lr)}")
            
        else:
            st.error("Unsupported problem type. Please select 'Regression' or 'Classification'.")
    
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")

def display_kpis(df, problem_type):
    """Display key performance indicators based on the problem type."""
    st.write("Key Performance Indicators:")
    if problem_type == "Regression":
        st.write("Example KPI 1: R-squared score, Mean Absolute Error, etc.")
    elif problem_type == "Classification":
        st.write("Example KPI 1: Accuracy, Precision, Recall, F1 Score, etc.")

def export_to_excel(df):
    """Export the DataFrame to an Excel file."""
    try:
        df.to_excel("results.xlsx", index=False)
        st.write("Results exported to Excel.")
    except Exception as e:
        st.error(f"Error exporting to Excel: {e}")

def create_plot(df, plot_type, x_col=None, y_col=None):
    """Create and return a plot based on the specified type and columns."""
    try:
        plt.figure(figsize=(10, 6))
        
        if plot_type == "scatter" and x_col and y_col:
            sns.scatterplot(data=df, x=x_col, y=y_col)
        elif plot_type == "line" and x_col and y_col:
            sns.lineplot(data=df, x=x_col, y=y_col)
        elif plot_type == "bar" and x_col and y_col:
            df.groupby(x_col)[y_col].mean().plot(kind="bar")
        elif plot_type == "histogram" and x_col:
            df[x_col].plot(kind="hist")
        else:
            st.write("Please specify the required columns for plotting.")
            return None

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error creating plot: {e}")
        return None

def main():
    """Main function to run the Streamlit app."""
    st.title("Automated ML Workflow with OpenAI Integration")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.subheader("Data Preview")
            st.write(df.head())
            
            if not df.empty:
                df = preprocess_data(df)
                st.subheader("Preprocessed Data")
                st.write(df.head())

                target_var = st.selectbox("Select the target variable", options=df.columns.tolist())
                perform_eda(df, target_var)
                df = feature_selection(df)

                x_vars = st.multiselect("Select independent variables", options=[col for col in df.columns if col != target_var])
                problem_type = st.selectbox("Select the type of problem", ["Regression", "Classification"])
                st.write(f"Selected Problem Type: {problem_type}")

                if st.button("Train Model"):
                    train_model(df, x_vars, target_var, problem_type)

                if st.button("Display KPIs"):
                    display_kpis(df, problem_type)

                st.write("Generate Insights using OpenAI")
                query = st.text_input("Enter your query for insights")
                if query:
                    insights = generate_description(query)
                    st.write(insights)

                st.write("Plotting Options")
                plot_type = st.selectbox("Select Plot Type", ["scatter", "line", "bar", "histogram"])
                x_col = st.selectbox("Select X-axis column", options=df.columns.tolist())
                y_col = st.selectbox("Select Y-axis column", options=df.columns.tolist())
                if st.button("Create Plot"):
                    buffer = create_plot(df, plot_type, x_col, y_col)
                    if buffer:
                        st.image(buffer)
                        
                if st.button("Export Results to Excel"):
                    export_to_excel(df)

if __name__ == '__main__':
    main()
