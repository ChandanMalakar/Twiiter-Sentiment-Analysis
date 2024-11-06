import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

# Initialize stemmer
port_stem = PorterStemmer()

# Load the trained model and vectorizer
with open("trained_model.sav", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.sav", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

st.title("Twitter Sentiment Analysis")

# User input for the tweet text
user_input = st.text_area("Enter the tweet text:")

# Predict sentiment on user input
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input text using the stemming function
        processed_input = stemming(user_input)
        
        # Transform the processed input text using the loaded vectorizer
        user_input_transformed = vectorizer.transform([processed_input])
        
        # Predict sentiment with probabilities (if model supports it)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input_transformed)
            confidence = np.max(proba)
            st.write(f"Prediction Confidence: {confidence:.2f}")

            # Set a threshold to determine "Neutral" if confidence is low
            if confidence < 0.6:
                st.write("Prediction: Neutral (Low Confidence)")
            else:
                prediction = model.predict(user_input_transformed)
                if prediction[0] == 0:
                    st.write("Prediction: Negative Sentiment")
                elif prediction[0] == 1:
                    st.write("Prediction: Positive Sentiment")
                else:
                    st.write("Prediction: Neutral Sentiment")
        else:
            # If model does not support predict_proba, use standard prediction
            prediction = model.predict(user_input_transformed)
            if prediction[0] == 0:
                st.write("Prediction: Negative Sentiment")
            elif prediction[0] == 1:
                st.write("Prediction: Positive Sentiment")
            else:
                st.write("Prediction: Neutral Sentiment")
    else:
        st.write("Please enter some text to predict the sentiment.")
