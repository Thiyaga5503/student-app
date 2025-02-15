import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

    def main():
        st.title("Student Performance prediction")
        st.write ("Enter your data to get a prediction for your performance")

        hour_studied = st.number_input("Hours studied ",min_value=1, max_value=10, value = 5 )
        Previou_score = st.number_input("previous score",min_value=40, max_value=100, value = 70 )
        Extra = st.selectbox("Extracurricular Activities ",['yes',"no"])
        sleeping_hours = st.number_input("Sleeping hours ",min_value=4, max_value=10, value = 7 )
        number_of_paper_solved = st.number_input("number of question paper solved",min_value=0, max_value=10, value = 5)

        if st.button("predict-your_score"):
            user_data = {
                "Hours_studied":hour_studied,
                "Previous Scores":previou_score,
                "Extracurricular Activities":extra,
                "Sleep hours":sleeping_hours,
                "number of papers solved":number_of_paper_solved
            }
            prediction = predict_data(user_data)
            st.success(f"your prediction result is {prediction}")

    if __name__ == "__main__":
        main()