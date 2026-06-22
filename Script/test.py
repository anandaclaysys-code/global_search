import joblib
import re
from nltk.stem import PorterStemmer
import pandas as pd
from evaluation_data import eva_data
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


data = eva_data()
print(f"Successfully loaded evaluation data. Number of samples: {len(data)}")
        # Show first 5 records as a DataFrame preview
df = pd.DataFrame(data)
print("\nPreview of the first 5 records:")
print(df.head())
correct_predictions = 0
total_predictions = len(data)
for i in range(len(data)):

    query = data[i]['query']
    actual_intent = data[i]['actual']
    preprocessed_query = preprocess(query)
    predicted_intent = loaded_pipe.predict([preprocessed_query])[0]
    # print(f"Query: '{query}' | Actual Intent: '{actual_intent}' | Predicted Intent: '{predicted_intent}'")
    if actual_intent == predicted_intent:
        correct_predictions += 1

print(f"\nCorrect Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {correct_predictions / total_predictions * 100:.2f}%")
    



# # Preprocess and predict on new queries
# new_queries = ["transfer money to savings", "pay my bill"]
# preprocessed_queries = [preprocess(q) for q in new_queries]
# predictions = loaded_pipe.predict(preprocessed_queries)

# for query, prediction in zip(new_queries, predictions):
#     print(f"Query: '{query}' -> Predicted Intent: '{prediction}'")
