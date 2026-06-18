import json
import re
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Initialize the stemmer
stemmer = PorterStemmer()

def stem_text(text: str) -> str:
    return ' '.join(stemmer.stem(word) for word in text.split())

stop_words = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'which', 'who', 'when', 'where', 'why', 'how'
}

def remove_stop_words(text: str) -> str:
    return ' '.join(word for word in text.split() if word not in stop_words)

def main():
    print("Loading data from SCCU-test.json...")
    with open('SCCU-test.json', encoding='utf-8') as f:
        raw = json.load(f)

    utts = raw['assets']['utterances']
    df = pd.DataFrame([{'text': u['text'], 'intent': u['intent']} for u in utts])

    print("Cleaning and preprocessing data...")
    df['text'] = df['text'].str.strip().str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # Drop rows with no alphanumeric content
    mask = df['text'].str.contains(r'[a-z0-9]', regex=True)
    df = df[mask].copy()

    # Drop duplicates
    df = df.drop_duplicates(subset=['text', 'intent']).reset_index(drop=True)

    # Remove stop words and stem
    df['text'] = df['text'].apply(remove_stop_words)
    df['text'] = df['text'].apply(stem_text)

    # Filter out classes with only 1 sample
    counts = df['intent'].value_counts()
    df = df[df['intent'].isin(counts[counts >= 2].index)].reset_index(drop=True)

    print(f"Dataset size for training: {len(df)} samples")

    # Define the pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm=None)),
        ('clf',   LogisticRegression(C=5, max_iter=1000, random_state=42))
    ])

    print("Fitting the pipeline...")
    pipe.fit(df['text'], df['intent'])

    print("Saving the fitted pipeline to 'intent_classifier.joblib'...")
    joblib.dump(pipe, 'intent_classifier.joblib')
    print("Model successfully trained and saved!")

if __name__ == '__main__':
    main()
