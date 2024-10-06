# 📚 Automated Machine Learning and NLP Workflow Application

This application provides an interactive and user-friendly interface for performing comprehensive data analysis, machine learning, and natural language processing tasks. 
Built with Streamlit, it streamlines the process of data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and inference.



## 🌟 Features

### 1. Supervised Learning
- **Data Upload**: Upload CSV datasets for analysis and modeling.
- **Data Preprocessing**:
  - Handle missing values and duplicates.
  - Encode categorical variables.
- **Exploratory Data Analysis (EDA)**:
  - **Univariate Analysis**: Histogram, Boxplot, Violinplot, KDE Plot.
  - **Bivariate Analysis**: Scatterplot, Heatmap, Boxplot, Bar Chart.
  - **Multivariate Analysis**: Pairplot, 3D Scatter Plot.
- **Feature Selection**:
  - Remove highly correlated features.
  - Select relevant features for modeling.
- **Model Training**:
  - **Classification Models**: Logistic Regression, Random Forest Classifier.
  - **Regression Models**: Linear Regression, Random Forest Regressor.
- **Model Evaluation**:
  - Display performance metrics such as Accuracy, Precision, Recall, F1-Score, MSE, MAE, R2 Score.
- **Visualization**:
  - Plot model performance and results.
- **Export Results**:
  - Save processed data and results to Excel files.

### 2. Natural Language Processing (NLP)
- **Data Upload**: Upload text data for NLP tasks.
- **Text Preprocessing**:
  - Tokenization using Hugging Face Tokenizers.
  - Stemming and Lemmatization using NLTK.
  - Removal of stopwords and punctuation.
- **Feature Extraction**:
  - TF-IDF Vectorization.
  - Bag of Words Model.
  - Word2Vec Embeddings.
- **Model Training**:
  - Fine-tune pre-trained models like BERT for text classification.
- **Model Evaluation**:
  - Evaluate NLP models using appropriate metrics.
- **Inference**:
  - Predict and interpret results on new text inputs.

### 3. Integration with Google Gemini API
- Generate insightful descriptions and summaries using the Google Gemini API based on the provided data and queries.




## 🔄 Workflow

### Supervised Learning Workflow
1. **Upload Dataset**: Users upload their CSV dataset through the application interface.
2. **Data Preprocessing**: The app automatically handles missing values and duplicates, preparing the data for analysis.
3. **Exploratory Data Analysis**:
   - Perform various analyses to understand data distributions and relationships.
   - Visualize data using different plot types.
4. **Feature Selection**: Identify and select the most relevant features by removing highly correlated or irrelevant features.
5. **Model Training**:
   - Choose the problem type (Classification or Regression).
   - Select appropriate models and train them on the processed data.
6. **Model Evaluation**:
   - Evaluate model performance using relevant metrics.
   - Visualize results and performance metrics.
7. **Export Results**: Save the processed data and model results for future reference or reporting.

### NLP Workflow
1. **Upload Text Data**: Users upload text data files for processing.
2. **Text Preprocessing**:
   - The app performs tokenization, stemming, lemmatization, and stopword removal.
3. **Feature Extraction**:
   - Extract features using TF-IDF, Bag of Words, or Word2Vec techniques.
4. **Model Training**:
   - Fine-tune pre-trained models (e.g., BERT) on the extracted features.
5. **Model Evaluation**:
   - Assess model performance using appropriate NLP metrics.
6. **Inference**:
   - Input new text data and obtain predictions from the trained model.
7. **Result Interpretation**:
   - Interpret and understand the predictions and model behavior.



## 🛠️ Tech Stack

- **Frontend & UI**:
  - [Streamlit](https://streamlit.io/): For building interactive web applications.
  
- **Data Manipulation & Analysis**:
  - [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
  - [NumPy](https://numpy.org/): For numerical computations.
  
- **Visualization**:
  - [Matplotlib](https://matplotlib.org/): For creating static, animated, and interactive visualizations.
  - [Seaborn](https://seaborn.pydata.org/): For statistical data visualization.
  
- **Machine Learning**:
  - [Scikit-Learn](https://scikit-learn.org/): For implementing machine learning algorithms.
  
- **Natural Language Processing**:
  - [NLTK](https://www.nltk.org/): For text preprocessing tasks like tokenization, stemming, and lemmatization.
  - [Transformers](https://huggingface.co/transformers/): For leveraging pre-trained models like BERT.
  - [Datasets](https://huggingface.co/docs/datasets): For handling and processing NLP datasets.
  
- **API Integration**:
  - [Google Gemini API](#): For generating descriptions and insights based on data.
  
- **Utilities**:
  - [Openpyxl](https://openpyxl.readthedocs.io/): For reading and writing Excel files.
  - [MPL Toolkits](https://matplotlib.org/mpl_toolkits/): For advanced plotting like 3D scatter plots.



## ⚙️ Installation

### Prerequisites
- Python 3.7 or higher installed on your system.
- Git installed for cloning the repository.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/automated-ml-nlp-app.git
   cd automated-ml-nlp-app
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   - **On Windows**
     ```bash
     venv\Scripts\activate
     ```
   - **On Unix or MacOS**
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Setup API Keys**
   - Replace `"Your_Google_Gemini_API_Key_Here"` with your actual Google Gemini API key in the code.
   - Ensure you have access and appropriate permissions for using the API.



## 🚀 Usage

1. **Run the Application**
   ```bash
   streamlit run app.py
   ```

2. **Access the App**
   - Open your web browser and navigate to `http://localhost:8501` to access the application.

3. **Using Supervised Learning Module**
   - Navigate to the **Supervised Learning** section via the sidebar.
   - **Upload Dataset**: Click on "Upload your CSV file" to upload your dataset.
   - **Data Overview**: View the first few rows and basic information about your dataset.
   - **EDA**: Perform exploratory data analysis by selecting the type of analysis and relevant features.
   - **Feature Selection**: Select features for model training and remove highly correlated features.
   - **Model Training**:
     - Choose the problem type (Classification or Regression).
     - Select features and target variable.
     - Click on "Train Model" to train and evaluate the model.
   - **Results**: View performance metrics and visualize results.
   - **Export Results**: Save the processed data and results as Excel files for future use.

4. **Using NLP Module**
   - Navigate to the **Natural Language Processing (NLP)** section via the sidebar.
   - **Upload Text Data**: Upload your text data file (e.g., CSV or TXT).
   - **Text Preprocessing**: The app automatically performs tokenization, stemming, lemmatization, and removes stopwords.
   - **Feature Extraction**: Features are extracted using methods like TF-IDF, Bag of Words, or Word2Vec.
   - **Model Training**: The app fine-tunes a pre-trained BERT model on your data.
   - **Model Evaluation**: View evaluation metrics such as accuracy, precision, recall, and F1-score.
   - **Inference**: Input new text and get predictions from the trained model.
   - **Result Interpretation**: Understand and interpret the model's predictions and performance.

 ---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "Add your message here"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

---

## 📞 Contact

- **Author**: ASHOK SAI G
- **Email**: aashoksai306@gmail.com
- **LinkedIn**: [ASHOK SAI G](https://www.linkedin.com/in/ashok-sai-g-165b98243?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BViXO8%2FSpRaGJBK4HdcGoSQ%3D%3D)

**Thank you for using the Automated Machine Learning and NLP Workflow Application!** 🎉

---
