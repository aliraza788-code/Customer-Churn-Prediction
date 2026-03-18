# Customer Churn Prediction (ML Pipeline)

##  Project Overview
This project predicts whether a customer will churn (leave) or not using Machine Learning.

##  Features
- End-to-End ML Pipeline using Scikit-learn
- Data preprocessing (scaling + encoding)
- Logistic Regression & Random Forest models
- Hyperparameter tuning using GridSearchCV
- Model selection based on accuracy
- Model export using joblib
- Streamlit web app for live predictions

##  Dataset
Telco Customer Churn Dataset

##  Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## ⚙️ How to Run

### 1. Install dependencies

pip install pandas scikit-learn streamlit joblib


### 2. Train model

python train.py


### 3. Run app

streamlit run app.py


##  Output
- Predicts whether customer will churn or not
- Displays result in web interface

##  Project Structure

churn-project/
│
├── train.py
├── app.py
├── model.pkl
├── telco.csv
└── README.md


##  Skills Gained
- ML Pipeline Construction
- Hyperparameter Tuning
- Model Deployment
- Production-ready ML system
