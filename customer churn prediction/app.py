import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Function to preprocess user input
def preprocess_input(input_data):
    # Convert categorical variables to numeric using LabelEncoder
    encoder = LabelEncoder()
    input_data['Geography'] = encoder.fit_transform([input_data['Geography']])
    input_data['Gender'] = encoder.fit_transform([input_data['Gender']])
    
    # Return preprocessed input as a DataFrame
    return pd.DataFrame(input_data)

# Function to predict churn based on user input
def predict_churn(input_data, model):
    # Preprocess the input data
    input_data_processed = preprocess_input(input_data)
    
    # Make predictions
    prediction = model.predict(input_data_processed)
    
    return prediction[0]

def main():
    # Load the trained model
    model_path = 'Gradient Boosting.pkl'  # Replace with your trained model path
    model = load_model(model_path)
    
    # Streamlit UI
    st.title('Customer Churn Prediction')
    st.write('Enter customer details to predict churn.')

    # User input fields
    credit_score = st.number_input('Credit Score', min_value=0, max_value=850, step=1)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    tenure = st.number_input('Tenure (years)', min_value=0, max_value=20, step=1)
    balance = st.number_input('Balance', min_value=0.0, step=1000.0)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0)
    
    # Predict button
    if st.button('Predict Churn'):
        user_input = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }
        
        # Predict churn based on user input
        prediction = predict_churn(user_input, model)
        
        # Display prediction
        if prediction == 1:
            st.error('The customer is predicted to churn.')
        else:
            st.success('The customer is predicted not to churn.')

if __name__ == '__main__':
    main()