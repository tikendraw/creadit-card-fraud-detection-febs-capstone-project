import pickle

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
import os
import joblib

# Load the trained model from a pickle file
def load_model(kind='joblib'):
    """ kind = 'joblib' or 'pkl' """
    
    pkl_link = "https://drive.google.com/file/d/1Bthv92aMCRYcIj80D8UIoHGgMFq3LO22/view?usp=sharing"
    joblib_link = "https://drive.google.com/file/d/1dmQfFcPQ503RgNkXuHxCbv5KT6CBXaI5/view?usp=sharing"
    
    pkl_id = pkl_link.split("/")[-2]
    joblib_id = joblib_link.split("/")[-2]
    
    if kind in ['joblib', 'jbl']:
        if not os.path.exists('./models/model_pipeline.joblib'):
            os.system(f"gdown --id {joblib_id} --output ./models/model_pipeline.joblib")

        with open('./models/model_pipeline.joblib', 'rb') as f:
            model = joblib.load(f)

        
    elif kind in ['pkl', 'pickle']:
        if not  os.path.exists('./models/model_pipeline.pkl'):
            os.system(f"gdown --id {pkl_id} --output ./models/model_pipeline.pkl")

        with open('./models/model_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
            
        

    print('+'*50)
    print(type(model))
    return model

model = load_model('joblib')

# Define the app
st.title('Credit Card Fraud Detection')

# Sidebar for single prediction
st.sidebar.header('Single Prediction')
input_data = {}

# Collect input data from the user
for col in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']:
    input_data[col] = st.sidebar.number_input(f'Input {col}', value=0.0)

# Convert user input to a DataFrame
input_df = pd.DataFrame([input_data])

# Make a prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.sidebar.write(f'Prediction: {prediction[0]}')
    st.sidebar.write(f'Probability: {prediction_proba[0]}')

# Batch prediction
st.header('Batch Prediction')
uploaded_file = st.file_uploader('Upload a CSV file for batch prediction( should have capitalized column names)', type=['csv'])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    batch_data.columns = [str(i).capitalize() for i in batch_data.columns]
    
    if col in batch_data.columns:
        if str(col).lower() == 'time':
            batch_data = batch_data.drop(col, axis=1)

    # Make predictions

    predictions = model.predict(batch_data)
    predictions_proba = model.predict_proba(batch_data)

    # Display results
    results_df = batch_data.copy()
    results_df['Prediction'] = predictions
    results_df['Probability'] = predictions_proba[:, 1]  # binary classification

    st.write(results_df)

    # Provide download link
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(results_df)

    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )
