import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

with open('trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.sav', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

from nltk.stem import PorterStemmer
import re

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

st.title("Twitter Sentiment Analysis")


user_input = st.text_area("Enter the tweet text:")


if st.button("Predict Sentiment"):
    if user_input:
        
        processed_input = stemming(user_input)
        user_input_transformed = vectorizer.transform([processed_input])
        
        
        prediction = model.predict(user_input_transformed)
        
        
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Prediction: {sentiment} Sentiment")
    else:
        st.write("Please enter some text to predict the sentiment.")
