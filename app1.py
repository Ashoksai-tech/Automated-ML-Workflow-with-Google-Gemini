import pandas as pd
import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D
import openpyxl

# Initialize the Gemini API client
genai.configure(api_key="AIzaSyDFqr07uAzPAB2ahk2ZmnahwX36x1E8gIA")

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

def generate_description(query, df):
    # Filter numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    # Missing values and duplicates
    missing_values_info = f"The dataset contains {df.isnull().sum().sum()} missing values."
    duplicates_info = f"The dataset contains {df.duplicated().sum()} duplicate rows."
    
    # Data accuracy and consistency
    accuracy_info = "Data accuracy and consistency checks should be based on domain knowledge and validation rules."
    
    # Outliers detection in numeric columns
    outlier_info = "Outliers detected in the following numeric columns: " + ", ".join(
        [col for col in numeric_df.columns if ((numeric_df[col] > numeric_df[col].quantile(0.99)) | (numeric_df[col] < numeric_df[col].quantile(0.01))).any()]
    )

    # Correlation between variables
    correlation_info = "The correlation matrix is:\n" + numeric_df.corr().to_string()
    
    # Relationships between variables
    relationship_info = "Relationships between variables can be explored through scatter plots, heatmaps, and pairplots."

    # Clusters or groups within the data
    clusters_info = "Clusters or groups can be identified using clustering algorithms like K-means."

    # Key features influencing the target variable
    key_features_info = "Key features influencing the target variable can be identified through feature importance analysis."

    # Visualization suggestions
    visualization_info = "Use histograms, scatter plots, and boxplots to understand data distribution and relationships."
    
    # Time series analysis (if applicable)
    time_series_info = "Time series analysis can be done using line plots and decompositions, if applicable."

    # Models to choose
    models_info = "Predictive models can be built based on the problem type: classification, regression, or unsupervised learning. Evaluate models based on their performance metrics such as accuracy, precision, recall, F1 score for classification, or RMSE for regression."
    
    # Model evaluation and improvement
    model_evaluation_info = "Performance of these models can be evaluated using metrics such as accuracy, precision, recall, F1 score, ROC AUC for classification, and RMSE, MAE for regression. Improving accuracy involves feature engineering, hyperparameter tuning, and using more sophisticated models."
    
    # Business problems and strategic decisions
    business_problems_info = "Business problems and strategic decisions depend on the specific use case of the data."
    strategic_decisions_info = "Strategic decisions can be guided by insights such as trend analysis and key performance indicators."
    
    # Key insights and business value
    insights_info = "Key insights include identifying trends, patterns, and actionable metrics that drive business value."

    # Combine all info into the prompt
    prompt = (
        f"Data collection method: {query}\n"
        f"{missing_values_info}\n"
        f"{duplicates_info}\n"
        f"{accuracy_info}\n"
        f"{outlier_info}\n"
        f"{relationship_info}\n"
        f"{correlation_info}\n"
        f"{clusters_info}\n"
        f"{key_features_info}\n"
        f"{visualization_info}\n"
        f"{time_series_info}\n"
        f"{models_info}\n"
        f"{model_evaluation_info}\n"
        f"{business_problems_info}\n"
        f"{strategic_decisions_info}\n"
        f"{insights_info}"
    )

    # Generate description with Gemini API
    insights = genai.generate_text(prompt=prompt)
    
    return insights.result

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

def plot_univariate(df, column, plot_type):
    """Generate and display a univariate plot."""
    try:
        # Create a figure and axis
        fig, ax = plt.subplots()

        if plot_type == "Histogram":
            df[column].hist(ax=ax)
        elif plot_type == "Boxplot":
            df.boxplot(column=column, ax=ax)
        elif plot_type == "Violinplot":
            import seaborn as sns
            sns.violinplot(x=df[column], ax=ax)
        elif plot_type == "KDE Plot":
            df[column].plot(kind='kde', ax=ax)
        else:
            st.error("Unknown plot type selected.")
            return

        # Display the plot in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error generating plot: {e}")

def plot_bivariate(df, column1, column2, plot_type):
    if plot_type == 'Heatmap':
        st.subheader(f'Correlation Heatmap between {column1} and {column2}')
        corr = df[[column1, column2]].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()
    elif plot_type == 'Scatterplot':
        st.subheader(f'Scatterplot between {column1} and {column2}')
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=df[column1], y=df[column2])
        st.pyplot()
    elif plot_type == 'Boxplot':
        st.subheader(f'Boxplot between {column1} and {column2}')
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column1], y=df[column2])
        st.pyplot()
    elif plot_type == 'Bar Chart':
        st.subheader(f'Bar Chart between {column1} and {column2}')
        df_grouped = df.groupby(column1).mean()[column2]
        df_grouped.plot(kind='bar')
        st.pyplot()

def plot_multivariate(df, plot_type):
    if plot_type == 'Pairplot':
        st.subheader('Pairplot of the dataset')
        sns.pairplot(df)
        st.pyplot()
    elif plot_type == '3D Scatter Plot':
        st.subheader('3D Scatter Plot of the dataset')
        if df.shape[1] >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.set_zlabel(df.columns[2])
            st.pyplot()
        else:
            st.error("Not enough columns for 3D scatter plot")

def feature_selection(df, selected_features):
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
    """Train a model based on the problem type and extract KPI metrics."""
    try:
        if y_var not in df.columns or not set(x_vars).issubset(df.columns):
            st.error("Invalid target or feature columns. Please check your input.")
            return

        X = df[x_vars]
        y = df[y_var]
        
        # Handle categorical features
        if X.select_dtypes(include=['object']).shape[1] > 0:
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if problem_type == "Classification":
            # Train a logistic regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Display metrics
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            # KPIs
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

        elif problem_type == "Regression":
            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Display metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"R2 Score: {r2:.2f}")

        else:
            st.error("Unsupported problem type. Please select Classification or Regression.")

    except Exception as e:
        st.error(f"Error training model: {e}")

def display_kpi(kpi_results, model_type):
    """Display Key Performance Indicators (KPIs) for the model."""
    st.subheader(f"{model_type} KPI Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{kpi_results['accuracy']:.2f}")
    with col2:
        st.metric("Precision", f"{kpi_results['precision']:.2f}")
    with col3:
        st.metric("Recall", f"{kpi_results['recall']:.2f}")
    with col1:
        st.metric("F1 Score", f"{kpi_results['f1']:.2f}")

def main():
    st.title("Data Analysis App")

    # Sidebar options
    sidebar_option = st.sidebar.selectbox("Select Analysis Type", ["Supervised Learning", "NLP"])

    if sidebar_option == "Supervised Learning":
        st.sidebar.subheader("Supervised Learning")
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

        df = load_data(uploaded_file)
        if df is not None:
            df = preprocess_data(df)
            st.write("Data Overview:")
            st.write(df.head())

            # EDA Options
            st.subheader("Exploratory Data Analysis (EDA)")

            analysis_type = st.selectbox("Select type of analysis", ["Univariate", "Bivariate", "Multivariate"])

            if analysis_type == "Univariate":
                column = st.selectbox("Select column for univariate analysis", df.columns)
                plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot", "Violinplot", "KDE Plot"])
                plot_univariate(df, column, plot_type)
                
            elif analysis_type == "Bivariate":
                column1 = st.selectbox("Select first column", df.columns)
                column2 = st.selectbox("Select second column", df.columns)
                plot_type = st.selectbox("Select plot type", ["Heatmap", "Scatterplot", "Boxplot", "Bar Chart"])
                plot_bivariate(df, column1, column2, plot_type)
                
            elif analysis_type == "Multivariate":
                plot_type = st.selectbox("Select plot type", ["Pairplot", "3D Scatter Plot"])
                plot_multivariate(df, plot_type)
                
            # Feature selection
            selected_features = st.multiselect("Select features", df.columns)
            df = feature_selection(df, selected_features)

            # Train Model
            st.subheader("Train Model")
            problem_type = st.selectbox("Select problem type", ["Classification", "Regression"])
            x_vars = st.multiselect("Select features", df.columns)
            y_var = st.selectbox("Select target variable", df.columns)

            if st.button("Train Model"):
                train_model(df, x_vars, y_var, problem_type)

    elif sidebar_option == "NLP":
        st.sidebar.subheader("Natural Language Processing (NLP)")

        st.write("NLP tools and options will be available here in future updates.")
        
if __name__ == "__main__":
    main()
