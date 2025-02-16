import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import pandas as pd
import streamlit as st
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder





def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.legend(loc=4)
    st.subheader("ROC Curve")
    st.pyplot(plt)

def train_model(X_train, X_test, y_train, y_test, model):


    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return accuracy, conf_matrix, cv_scores, y_pred_proba, model



def perform_feature_engineering(data, target_variable):


    st.subheader("Feature Engineering")

    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        data = pd.get_dummies(data, columns=categorical_columns)
    
    imputer = SimpleImputer(strategy="mean")
    data.iloc[:, :] = imputer.fit_transform(data)
    feature_selection_method = st.selectbox("Select Feature Selection Method", ["None", "Random Forest Importance", "Chi-Square Test"])
    
    selected_features = data.columns.tolist()
    
    if feature_selection_method != "None":
        X = data.drop(columns=[target_variable])
        y = data[target_variable
                 ]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
            data[target_variable] = LabelEncoder().fit_transform(data[target_variable])
        
        if feature_selection_method == "Random Forest Importance":
            model = RandomForestClassifier()
            model.fit(X, y)
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.write("Feature Importances:")
            st.bar_chart(feature_importances)
            num_features = min(8, len(feature_importances))
            selected_features = feature_importances.nlargest(num_features).index.tolist()
        
        elif feature_selection_method == "Chi-Square Test":
            chi_selector = SelectKBest(chi2, k=min(5, len(X.columns)))
            chi_selector.fit(X, y)
            selected_features = X.columns[chi_selector.get_support()].tolist()
            st.write("Chi-Square Scores:")
            st.bar_chart(pd.Series(chi_selector.scores_, index=X.columns).sort_values(ascending=False))

    new_data = data[selected_features + [target_variable]]


    
    return new_data, selected_features,imputer


def performe_feature_engineering_prediction(data,imputer,selected_features):
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        # st.write("Encoding categorical variables...")
        data = pd.get_dummies(data, columns=categorical_columns)

    imputer = SimpleImputer(strategy="mean")
    data.iloc[:, :] = imputer.fit_transform(data)  # Apply imputation to the whole dataframe

    return data[selected_features]