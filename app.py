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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classification Web App')
    st.markdown('Are your mushrooms edible or poisonous? 🐷')
    st.sidebar.markdown('Are your mushrooms edible or poisonous? 😃')

    @st.cache(persist=True)  # cache the output to disk
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
        c = st.sidebar.number_input("C (Regularization parameter 0.01-10.0)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        # Train the model and print the results
        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=c, kernel=kernel, gamma=gamma)
            model.fit(x1, y1)
            # model = make_pipeline(StandardScaler(), SVC(C=c, kernel=kernel, gamma=gamma))
            # model.fit(x1, y1)
            accuracy = model.score(x2, y2)
            y_pred = model.predict(x2)
            precision_score_xy = precision_score(y2, y_pred, labels=class_names)
            recall_score_xy = recall_score(y2, y_pred, labels=class_names)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score_xy, 2))
            st.write("Recall: ", round(recall_score_xy, 2))

            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        c_lr = st.sidebar.number_input("C (Regularization parameter 0.01-10.0)", 0.01, 10.0, step=0.01, key='C_lr')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        # Train the model and print the results
        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=c_lr, max_iter=max_iter)
            model.fit(x1, y1)
            accuracy = model.score(x2, y2)
            y_pred = model.predict(x2)
            precision_score_xy = precision_score(y2, y_pred, labels=class_names)
            recall_score_xy = recall_score(y2, y_pred, labels=class_names)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score_xy, 2))
            st.write("Recall: ", round(recall_score_xy, 2))

            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest (100-5000)",
                                               100, 5000, step=100, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree (1-20)",
                                            1, 20,step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",
                                     ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        # Train the model and print the results
        if st.sidebar.button('Classify', key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,
                                           bootstrap=bootstrap,
                                           max_depth=max_depth,
                                           n_jobs=-1)
            model.fit(x1, y1)
            accuracy = model.score(x2, y2)
            y_pred = model.predict(x2)
            precision_score_xy = precision_score(y2, y_pred, labels=class_names)
            recall_score_xy = recall_score(y2, y_pred, labels=class_names)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score_xy, 2))
            st.write("Recall: ", round(recall_score_xy, 2))

            plot_metrics(metrics)
    # set the default state of Checkbar to be unchecked
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
