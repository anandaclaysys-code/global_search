import json
import bm25s

# 1. Load your dataset (assuming it is saved as 'dataset.json')
# Alternatively, replace this with your raw dictionary object
with open(r'C:\Users\anandha.kumar\OneDrive - ClaySys Technologies\Desktop\Code Base\Global search\data\processed\restructured_dataset.json', 'r') as f:
    dataset_json = json.load(f)

# 2. Flatten the dataset into a corpus and map to corresponding intents
corpus = []
intent_mapping = []

for intent, phrases in dataset_json.items():
    for phrase in phrases:
        corpus.append(phrase)
        intent_mapping.append(intent)

# 3. Simple tokenization helper (lowercasing and splitting by whitespace)
# BM25-S expects an array of tokenized strings
def tokenize(text):
    return text.lower().split()

corpus_tokens = [tokenize(doc) for doc in corpus]

# 4. Initialize and index the BM25 model
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# 5. Define the Prediction Function
def predict_intent(query, top_n=3):
    # Tokenize the incoming query
    query_tokens = tokenize(query)
    
    # Retrieve top match document indices and scores
    # corpus_tokens needs to be wrapped in a list if querying a single item
    results = retriever.retrieve([query_tokens], k=top_n)
    
    # Extract results
    matched_doc_index = results.documents[0][0]
    score = results.scores[0][0]
    
    # Fallback if no keywords matched (score is 0)
    if score == 0:
        return "Unknown / Out of Domain", 0.0
        
    predicted_intent = intent_mapping[matched_doc_index]
    return predicted_intent, score

# --- 6. Test Queries ---
test_queries = [
    "I want to pay my electricity bill",
    "where is my check deposit history?",
    "how to update my email address"
]

print("--- Intent Prediction Results ---")
for q in test_queries:
    intent, score = predict_intent(q)
    print(f"Query: '{q}' -> Predicted Intent: **{intent}** (Score: {score:.4f})")

with open(r"C:\Users\anandha.kumar\OneDrive - ClaySys Technologies\Desktop\Code Base\Global search\data\raw\evaluation_dataset.json") as f:
    evaluation_dataset = json.load(f)

# 7. Evaluate the model
test_queries_evaluation=evaluation_dataset['test_queries']      
print(test_queries_evaluation)