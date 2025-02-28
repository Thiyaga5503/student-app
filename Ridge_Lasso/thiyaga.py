import streamlit as st  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pickle
import time
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from pymongo.mongo_client import MongoClient  # type: ignore
from pymongo.server_api import ServerApi  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# # Retrieve credentials securely from Streamlit Secrets
# mongo_uri = st.secrets["mongodb"]["uri"]
# db_name = st.secrets["mongodb"]["database"]
# collection_name = st.secrets["mongodb"]["collection"]

# # Connect to MongoDB
# client = MongoClient(mongo_uri, server_api=ServerApi('1'))
# database = client[db_name]
# collection = database[collection_name]


# Function to load the model
def load_model(model_name):
   
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function to process input data
def processing_input_data(data, scaler):
    
    data = pd.DataFrame([data])
    data["sex"] = data["sex"].map({"Male": 1, "Female": 2})
    data_transformed = scaler.transform(data)
    return data_transformed

# Function for prediction
def predict_data(data, model_name):
    model= load_model(model_name)
    processed_data = processing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction
    


# Main Streamlit App
def main():
  
    st.set_page_config(page_title="🔬 Diabetes Predictor", page_icon="🩺", layout="wide")

    st.title("🚀 **AI-Based Diabetes Progression Prediction**")
    st.markdown("🔬 **Using Machine Learning to Estimate Diabetes Progression Over Time**")

    # 🎨 Custom Background Styling
    st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ffefba, #ffffff);
        }
        div.stButton > button:first-child {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    # Sidebar Layout
    st.sidebar.header("⚙️ **Configuration**")
    st.sidebar.write("Select the prediction model.")

    model_choice = st.sidebar.radio("Choose Model", ["🤖 Ridge Regression", "🧮 Lasso Regression"])

    st.sidebar.markdown("---")  # Add a divider

    st.sidebar.header("📋 **Patient Data Input**")
    st.sidebar.write("Fill in the details below to predict diabetes progression.")
    
    age = st.sidebar.slider("🎂 Age of Patient", 18, 80, 25)
    sex = st.sidebar.selectbox("⚤ Sex of Patient", ["Male", "Female"])
    bmi = st.sidebar.slider("⚖️ BMI of Patient", 18.0, 43.0, 25.0)
    bp = st.sidebar.slider("💉 Blood Pressure", 60, 180, 120)
    s1 = st.sidebar.slider("🩸 Total Serum Cholesterol", 90, 400, 200)
    s2 = st.sidebar.slider("🧪 Low-Density Lipoproteins (LDL)", 50, 250, 100)
    s3 = st.sidebar.slider("💊 High-Density Lipoproteins (HDL)", 20, 100, 50)
    s4 = st.sidebar.slider("🔗 Total Cholesterol / HDL Ratio", 1.5, 10.0, 4.5)
    s5 = st.sidebar.slider("🩺 Log of Serum Triglycerides", 3.0, 6.5, 5.2)
    s6 = st.sidebar.slider("🩸 Blood Sugar Level", 50, 600, 99)

    user_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "bp": bp,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "s6": s6
    }

    if st.sidebar.button("🚀 Predict"):
        with st.spinner("🕒 Processing your input... Please wait"):
            time.sleep(2)  # Simulate a loading delay
         
            # Map Model Selection
            model_name = "Ridge_model.pkl" if model_choice == "🤖 Ridge Regression" else "Lasso_model.pkl"
            model_Id = "Ridge" if model_choice == "🤖 Ridge Regression" else "Lasso"

            # Make Prediction
            prediction = predict_data(user_data, model_name)
             
            # Store user data in MongoDB
            user_data["quantitative measure of disease progression"] = float(prediction[0])
            user_data["model_name"] = model_Id
            
        #
            
            #collection.insert_one(document)
        
        # Display Results
        st.markdown(f"## 🎯 Prediction Result")
        st.success(f"📊 **Estimated Disease Progression Score: {prediction[0]:.2f}**")

        
        



if __name__ == "__main__":
    # Run the Streamlit app
    main()