import joblib
import re
from nltk.stem import PorterStemmer

# Load the model
loaded_pipe = joblib.load('intent_classifier.joblib')

# Initialize stemmer and stop words matching the training phase
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

# Preprocess and predict on new queries
new_queries = ["transfer money to savings", "pay my bill"]
preprocessed_queries = [preprocess(q) for q in new_queries]
predictions = loaded_pipe.predict(preprocessed_queries)

for query, prediction in zip(new_queries, predictions):
    print(f"Query: '{query}' -> Predicted Intent: '{prediction}'")
