import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Thiyaga:1234@cluster0.zpln3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['SVM']
collection = db["svm_pred"]

# Load the iris dataset
data = load_iris()

# Create a DataFrame from the dataset
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['target_names'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})

# Create a binary classification dataset
binary_df = df[df['target'] != 2]
x_binary = binary_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y_binary = binary_df['target']

# Split the binary dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_binary, y_binary, test_size=0.2)

# Scale the training and testing sets
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train a binary SVM classifier
svm_binary = SVC(kernel='rbf', C=60.0, probability=True)
svm_binary.fit(x_train, y_train)

# Train a binary logistic regression classifier
logistics_binary = LogisticRegression()
logistics_binary.fit(x_train, y_train)

# Create a multi-class classification dataset
x_multi = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y_multi = df['target']

# Split the multi-class dataset into training and testing sets
x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(x_multi, y_multi, test_size=0.2)

# Train a multi-class SVM classifier
svm_multi = SVC(kernel="linear")
svm_multi.fit(x_train_multi, y_train_multi)

# Train a multi-class logistic regression classifier with one-vs-rest strategy
logistics_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=2000)
logistics_ovr.fit(x_train_multi, y_train_multi)

# Train a multi-class logistic regression classifier with multinomial strategy
logistics_multinomial = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
logistics_multinomial.fit(x_train_multi, y_train_multi)

# Create a Streamlit app
def main():
 st.title("Classification Models")

# Display the binary classification results
st.header("Binary Classification")
st.subheader("SVM Classifier")
st.write("Classification Report:")
st.write(classification_report(y_test, svm_binary.predict(x_test)))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, svm_binary.predict(x_test)))

st.subheader("Logistic Regression Classifier")
st.write("Classification Report:")
st.write(classification_report(y_test, logistics_binary.predict(x_test)))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, logistics_binary.predict(x_test)))

# Display the multi-class classification results
st.header("Multi-Class Classification")
st.subheader("SVM Classifier")
st.write("Classification Report:")
st.write(classification_report(y_test_multi, svm_multi.predict(x_test_multi)))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test_multi, svm_multi.predict(x_test_multi)))

st.subheader("Logistic Regression Classifier (One-vs-Rest)")
st.write("Classification Report:")
st.write(classification_report(y_test_multi, logistics_ovr.predict(x_test_multi)))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test_multi, logistics_ovr.predict(x_test_multi)))

st.subheader("Logistic Regression Classifier (Multinomial)")
st.write("Classification Report:")
st.write(classification_report(y_test_multi, logistics_multinomial.predict(x_test_multi)))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test_multi, logistics_multinomial.predict(x_test_multi)))

if __name__ == "__main__":
    main()