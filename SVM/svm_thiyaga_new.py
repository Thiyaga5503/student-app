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
from pymongo.mongo_client import MongoClient  # type: ignore
from pymongo.server_api import ServerApi  # type: ignore

uri = "mongodb+srv://Thiyaga:1234@cluster0.zpln3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['IrisClassification']
collection = db["IrisPrediction"]

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
    st.title("Iris Classification App")

    # Input form
    st.header("Enter Input Values")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)

    # Predict button
    if st.button("Predict"):
        # Create a new input array
        new_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale the new input array
        new_input_scaled = scaler.transform(new_input)

        # Predict the result using the binary SVM classifier
        prediction_binary = svm_binary.predict(new_input_scaled)

        # Predict the result using the binary logistic regression classifier
        prediction_logistics_binary = logistics_binary.predict(new_input_scaled)

        # Predict the result using the multi-class SVM classifier
        prediction_multi = svm_multi.predict(new_input)

        # Predict the result using the multi-class logistic regression classifier with one-vs-rest strategy
        prediction_logistics_ovr = logistics_ovr.predict(new_input)

        # Predict the result using the multi-class logistic regression classifier with multinomial strategy
        prediction_logistics_multinomial = logistics_multinomial.predict(new_input)

        # Display the predictions
        st.header("Predictions")
        st.write("Binary SVM Classifier:", data.target_names[prediction_binary[0]])
        st.write("Binary Logistic Regression Classifier:", data.target_names[prediction_logistics_binary[0]])
        st.write("Multi-Class SVM Classifier:", data.target_names[prediction_multi[0]])
        st.write("Multi-Class Logistic Regression Classifier (One-vs-Rest):", data.target_names[prediction_logistics_ovr[0]])
        st.write("Multi-Class Logistic Regression Classifier (Multinomial):", data.target_names[prediction_logistics_multinomial[0]])

 # Store the predictions in the MongoDB database
    prediction_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
        "binary_svm_prediction": data.target_names[prediction_binary[0]],
        "binary_logistics_prediction": data.target_names[prediction_logistics_binary[0]],
        "multi_svm_prediction": data.target_names[prediction_multi[0]],
        "multi_logistics_ovr_prediction": data.target_names[prediction_logistics_ovr[0]],
        "multi_logistics_multinomial_prediction": data.target_names[prediction_logistics_multinomial[0]]
    }
    collection.insert_one(prediction_data)

if __name__ == "__main__":
    main()
