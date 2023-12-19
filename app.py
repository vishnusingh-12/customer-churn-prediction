import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Churn Predictor")

# brand
location = st.selectbox('Location', df['Location'].unique())

# type of laptop
gender = st.selectbox('Gender', df['Gender'].unique())

# Ram
subscription_length = st.selectbox('Subscription (Months)', [i for i in range(1, 25)])

# weight
age = st.number_input('Age')
bill = st.number_input('Monthly Bill')
usage = st.number_input('Total Data Usage(GB)')

if st.button('Predict'):
    # query
    churn = {0: 'not churn', 1: 'churn'}
    query = np.array([age, gender, location, subscription_length, bill, usage])

    query = query.reshape(1, 6)
    st.title("This customer might " + churn[(pipe.predict(query)[0])])
