import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle


#Load the trained model
model=tf.keras.models.load_model('churn_model.h5')

#Load the encoder and scaler
with open('onehotencoder_geo.pkl', 'rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:   
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

##StreamLIit App
st.title('Customer Churn Prediction')

#user input
geography = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
numOfProducts=st.slider('Number of Products',1,4)
hasCrCard=st.selectbox('Has Credit Card',[0,1])
isActiveMember=st.selectbox('Is Active Member',[0,1])


#Prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numOfProducts],
    'HasCrCard': [hasCrCard],
    'IsActiveMember': [isActiveMember],
    'EstimatedSalary': [estimated_salary]
})

#One-hot encode the geography
geo_encoded = onehotencoder_geo.transform(np.array(input_data['Geography']).reshape(-1,1)).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehotencoder_geo.get_feature_names_out(['Geography']))

#Concatenate the data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


#scale the input data 
input_data_scaled=scaler.transform(input_data)


#Predict the churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

#Display the prediction
st.write(f'Churn Probability: {prediction_proba:.2f}')
if prediction_proba>=0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is likely to stay')