import streamlit as st
import joblib
import re
import os
from nltk.stem import PorterStemmer

# Set page config
st.set_page_config(page_title="Intent Predictor", page_icon="🔍")

# Stemmer and stop words matching the training phase
stemmer = PorterStemmer()
stop_words = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'which', 'who', 'when', 'where', 'why', 'how'
}

def preprocess(query: str) -> str:
    # Lowercase and clean whitespace
    clean_query = query.lower().strip()
    clean_query = re.sub(r'\s+', ' ', clean_query)
    # Remove stop words
    clean_query = ' '.join(word for word in clean_query.split() if word not in stop_words)
    # Stem words
    clean_query = ' '.join(stemmer.stem(word) for word in clean_query.split())
    return clean_query

# Load the model
model_path = 'intent_classifier.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    st.error("Model file 'intent_classifier.joblib' not found. Please train the model first.")

# Title
st.title("🔍 Intent Predictor")
st.write("Enter your query below to identify its intent.")

# Text input box
query = st.text_input("Enter Query:", placeholder="Type something here...")

# Predict intent
if query:
    if model is not None:
        preprocessed_query = preprocess(query)
        prediction = model.predict([preprocessed_query])[0]
        
        # Display the result
        st.success(f"**Predicted Intent:** {prediction}")
