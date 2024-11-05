import streamlit as st
import pickle
import time
from Pre import Preprocessing
import pandas as pd

with open('/Users/pramodchoudhary/Documents/Program/SEM-2/IRT Project/Sentiment_Analysis_model.pkl', 'rb') as f:
    model,Preprocessing = pickle.load(f)

st.title('Twitter Sentiment Analysis')

tweet = st.text_input('Enter your tweet')

submit = st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict(pd.Series(tweet))
    end = time.time()

    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    
    if prediction[0]==0:
        print('Negative')
        st.write('Negative')
    else:
        print('Positive')
        st.write('Positive')



# with st.form(key='prediction_form'):
#     tweet = st.text_input('Enter your tweet')

#     submit_button = st.form_submit_button(label='Predict')

#     if submit_button:
#         start = time.time()
#         prediction = model.predict(pd.Series(tweet))
#         end = time.time()

#         st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
        
#         if prediction[0]==0:
#             print('Negative')
#             st.write('Negative')
#         else:
#             print('Positive')
#             st.write('Positive')