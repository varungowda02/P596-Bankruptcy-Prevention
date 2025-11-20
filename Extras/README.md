# Bankruptcy Predictor

This Streamlit web app predicts bankruptcy risk using a pre-trained machine learning model.

## Features
- Upload CSV or enter financial indicators manually  
- Predict bankruptcy risk (Low / Medium / High)  
- Download prediction as PDF report  

## Deployment
Deployed on Streamlit Cloud.  
Trained model: `bankruptcy_model.joblib`  
Dataset used: `Bankruptcy.xlsx`

bankruptcy-predictor/
│
├── bankruptcy_predict.py         ← Streamlit app 
├── bankruptcy_model.joblib       ← saved ML model
├── Bankruptcy.xlsx               ← dataset (for loading df)
├── requirements.txt              ← dependencies file
└── README.md                     ← short project description
