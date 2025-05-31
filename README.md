# AutoML - Automatic Machine Learning

A web application that automates the machine learning workflow, from data exploration to model training and prediction.

## Features

- **Exploratory Data Analysis**: Generate comprehensive EDA reports using Sweetviz
- **Model Training**: Choose from multiple algorithms:
  - Random Forest
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine
  - Gradient Boosting
- **Feature Engineering**:
  - Automated handling of categorical variables
  - Missing value imputation
  - Feature selection methods:
    - Random Forest Importance
    - Chi-Square Test
- **Prediction**: Upload new data to get predictions from trained models
- **Visualization**: View model performance metrics and prediction results

## Requirements

- Python 3.x
- Pandas
- Streamlit
- Scikit-learn
- Sweetviz
- Matplotlib

## Usage

1. Clone the repository
2. Install dependencies:
   ```
   pip install pandas streamlit scikit-learn sweetviz matplotlib
   ```
3. Run the application:
   ```
   streamlit run newml.py
   ```

## Workflow

1. Upload your CSV dataset
2. Select the target variable
3. Choose between EDA or model training
4. For EDA: Download the generated report
5. For model training:
   - Select feature engineering options
   - Choose a machine learning algorithm
   - Review model performance
   - Upload new data for predictions
   - Download prediction results

## Deployment

The application includes configuration for deployment on platforms like Heroku using the provided Procfile and setup.sh files.
