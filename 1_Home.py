import pandas as pd
import streamlit as st
import numpy as np
import pickle
import math

st.set_page_config(page_title = 'Abalone Seafood Farming', page_icon=':shell:', layout='wide')
st.write("""
# Abalone Age Prediction
Helping the farmers estimate the age of the abalone using its physical characteristics with machine learning
""")

def user_input_features():
    sex = st.selectbox('What is your abalone gender?', ['Male', 'Female', 'Infant'])
    length = st.number_input('what is your abalone longest shell measurement?', 0, 100000)
    diameter =  st.number_input('What is your abalone perpendicular to the length?', 0, 1000000000)
    height = st.number_input('What is your abalone height?', 0, 1000000000)
    whole_wt = st.number_input('What is your whole abalone weight?', 0, 1000000000)
    shucked_wt = st.number_input('What is your abalone meat weight?', 0, 1000000000)
    viscera_wt = st.number_input('What is your abalone gut-weight?', 0, 1000000000)
    shell_wt = st.number_input('What is your abalone dried shell weight?', 0, 1000000000)
    sex = 'M' if sex == 'Male' else 'F' if sex == 'Female' else 'I'
    data = {'sex': sex,
            'length': length,
            'diameter': diameter,
            'height': height,
            'whole_wt': whole_wt,
            'shucked_wt': shucked_wt,
            'viscera_wt': viscera_wt,
            'shell_wt': shell_wt}
    features = pd.DataFrame(data, index = [0])
    return features

input_df = user_input_features()

abalone = pd.read_csv('datasets/abalone.csv')

def predict():
    X_test = input_df.copy()

    # Data Preprocessing
    X_test['trans_length'] = (X_test['length']) ** 2
    X_test['trans_diameter'] = X_test['diameter'] ** 2

    for num in ['height', 'whole_wt', 'shucked_wt', 'viscera_wt', 'shell_wt']:
        X_test['trans_'+num] = np.sqrt(X_test[num])

    # One Hot Encoding
    onehots = pd.get_dummies(abalone['sex'], prefix = 'sex')
    onehots.loc[0, :] = 0
    val = onehots.head(1)
    if ('sex_'+X_test['sex'].values[0]) in val.columns:
        val['sex'+'_'+X_test['sex'].values] = 1
    X_test.drop(columns = 'sex', inplace = True)
    X_test = X_test.join(val)

    # Drop Unnecessary Features
    X_test.drop(columns = ['length', 'diameter','height', 'whole_wt', 'shucked_wt', 'viscera_wt', 'shell_wt'], inplace = True)
    # Scale Test set
    load_scaler = pickle.load(open('scaler_reg.pkl', 'rb'))
    X_test_scaled = load_scaler.transform(X_test)

    # Predict Scaled test set with decision tree regression model
    load_model = pickle.load(open('model_reg.pkl', 'rb'))
    prediction = load_model.predict(X_test_scaled)[0]
    result = (math.e) ** prediction
    st.success(f'Your abalone age is {round(result, 1)} years old')

st.button('Predict', on_click=predict)




