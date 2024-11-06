import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the trained model and vectorizer
with open('trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.sav', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Stemming function
from nltk.stem import PorterStemmer
import re

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

st.title("Twitter Sentiment Analysis")

# User input
user_input = st.text_area("Enter the tweet text:")

# Prediction button
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess input
        processed_input = stemming(user_input)
        user_input_transformed = vectorizer.transform([processed_input])
        
        # Predict sentiment
        prediction = model.predict(user_input_transformed)
        
        # Display the prediction
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Prediction: {sentiment} Sentiment")
    else:
        st.write("Please enter some text to predict the sentiment.")
