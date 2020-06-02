import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Are your mushrooms edible or poisonous? 🐷')
    st.sidebar.markdown('Are your mushrooms edible or poisonous? 😃')

    @st.cache(persist=True)# cache the output to disk
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        # instantiate helper function
        label = LabelEncoder()
        # Iterate every columns in our data frame and fit transform
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(dataframe):
        y = dataframe.type
        x = dataframe.drop(columns=['type'])
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x2, y2, display_labels=class_names)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x2, y2)
            st.pyplot()
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x2, y2)
            st.pyplot()

    # store data into a Dataframe
    df = load_data()
    x1, x2, y1, y2 = split(df)
    class_names = ['Edible', 'Poisonous']

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Super Vector Machine (SVM)",
         "Logistic Regression",
         "Random Forest"))

    if classifier == 'Super Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, steps=0.01, key = 'C')
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key='kernel')












    # set the default state of Checkbar to be unchecked
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

if __name__ == '__main__':
    main()
