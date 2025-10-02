import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("/Users/shyamp_1/Documents/Documents - Shyam’s MacBook Air/GitHub/ML-Text-Detector/spam.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess, cat, test_size = 0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#Creating Model

model = MultinomialNB()
model.fit(features, cat_train)

#Test our model

features_test = cv.transform(mess_test)

#Predict Data in Real-Time

def predict(message):
    x = cv.transform([message])
    y = model.predict(x)[0]
    return str(y)

st.header('Spam Detection')
input_mess = st.text_input('Enter Message Here')
if st.button('Validate') and input_mess:
    output = predict(input_mess)
    st.success(f'Prediction: {output}')


