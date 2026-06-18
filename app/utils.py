import streamlit as st
import joblib
import re
import os
import json
import pandas as pd
from nltk.stem import PorterStemmer

# Stemmer and stop words matching the training phase
@st.cache_resource
def load_stemmer_and_stopwords():
    """Caches and loads the Porter stemmer and a set of custom stop words."""
    stemmer = PorterStemmer()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'which', 'who', 'when', 'where', 'why', 'how'
    }
    return stemmer, stop_words

# Load stemmer and stopwords at module level
stemmer, stop_words = load_stemmer_and_stopwords()

def preprocess(query: str) -> str:
    """Preprocesses a query by lowercasing, removing stop words, and stemming."""
    # Lowercase and clean whitespace
    clean_query = query.lower().strip()
    clean_query = re.sub(r'\s+', ' ', clean_query)
    # Remove stop words
    clean_query = ' '.join(word for word in clean_query.split() if word not in stop_words)
    # Stem words
    clean_query = ' '.join(stemmer.stem(word) for word in clean_query.split())
    return clean_query

# Load classifier model
@st.cache_resource
def load_model(model_path: str):
    """Caches and loads the trained classification model pipeline from path."""
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

# Load dataset metadata if exists
@st.cache_data
def get_dataset_info(json_path: str):
    """Loads and caches dataset properties like utterance and intent count."""
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)
            utts = data['assets']['utterances']
            df = pd.DataFrame([{'text': u['text'], 'intent': u['intent']} for u in utts])
            intents = df['intent'].unique()
            return len(df), len(intents), list(intents)
        except Exception:
            pass
    return None, None, None
