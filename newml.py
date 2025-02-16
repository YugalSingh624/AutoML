import pandas as pd
import streamlit as st
import base64

import sweetviz as sv
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import streamlit as st

from utils import train_model,plot_roc_curve,perform_feature_engineering,performe_feature_engineering_prediction





def main():
    st.markdown("""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Automatic Machine Learning</h2>
    </div>
    """, unsafe_allow_html=True)
    
    file_upload = st.sidebar.file_uploader("Upload Input CSV File", type=["csv"])
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        
        if st.checkbox("Show Input Data"):
            st.write(data)
        
        target_variable = st.selectbox("Select Target Variable", options=["None"] + list(data.columns))
        if target_variable != "None":

        
            analysis_option = st.selectbox("Choose an option:", options=["None","Perform EDA", "Train a Model"])
            
            if analysis_option == "Perform EDA":
                st.subheader("Exploratory Data Analysis Report")
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                report = sv.analyze(data)
                report.show_html("eda_report.html")
                with open("eda_report.html", "rb") as file:
                    b64 = base64.b64encode(file.read()).decode()
                    href = f'<a href="data:file/html;base64,{b64}" download="EDA_Report.html">Download EDA Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            elif analysis_option == "Train a Model":
                
                data, selected_features,imputer= perform_feature_engineering(data, target_variable)
                feature_columns = [col for col in data.columns if col != target_variable]
                X = data[feature_columns]
                y = data[target_variable]


                st.subheader("Pick Your Algorithm")
                model_options = {
                    'Random Forest': RandomForestClassifier(),
                    'Logistic Regression': LogisticRegression(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'Support Vector Machine': SVC(probability=True),
                    'Gradient Boosting': GradientBoostingClassifier()
                }
                
                chosen_model_name = st.selectbox("Select Model", options=["None"] + list(model_options.keys()))
                if chosen_model_name != "None":
                    model = model_options[chosen_model_name]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                    
                    accuracy, conf_matrix, cv_scores, y_pred_proba, model = train_model(X_train, X_test, y_train, y_test, model)
                    
                    st.write("Model Accuracy:", accuracy)
                    st.write("Confusion Matrix:", conf_matrix)
                    st.write("Cross Validation Scores:", cv_scores)
                    
                    if y_pred_proba is not None:
                        plot_roc_curve(y_test, y_pred_proba)
                    
                    st.subheader("Upload CSV file for Predictions:")
                    file_upload = st.file_uploader(" ", type=["csv"])

                    if file_upload is not None:
                        original_data = pd.read_csv(file_upload)  # Load original data
                        
                        encoded_data = performe_feature_engineering_prediction(original_data,imputer,selected_features)

                        # Predict
                        predictions = model.predict(encoded_data)

                        # Restore original format
                        original_data['Prediction'] = predictions

                            

                        st.subheader("Find the Predicted Results below:")
                        st.write(original_data)

                        st.text(f"0 : Not {target_variable}")
                        st.text(f"1 : {target_variable}")

                        # Convert to CSV for download (original format)
                        csv = original_data.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode() # Encode CSV to base64
                        href = f'<a href="data:file/csv;base64,{b64}" download="Predicted_Results.csv">Download The Prediction Results CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Visualization of predicted values
                        display_df = st.checkbox(label='Visualize the Predicted Values')
                        if display_df:
                            st.bar_chart(original_data['Prediction'].value_counts())
                            st.text(original_data['Prediction'].value_counts())



if __name__ == '__main__':
    main()