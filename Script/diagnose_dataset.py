import os
import json
import pandas as pd
import joblib
from collections import Counter
from nltk.stem import PorterStemmer

# Define paths
WORKSPACE_DIR = r"C:\Users\anandha.kumar\OneDrive - ClaySys Technologies\Desktop\Code Base\Global search"
EVAL_DATASET_PATH = os.path.join(WORKSPACE_DIR, "data", "raw", "evaluation_dataset.json")
MODEL_PATH = os.path.join(WORKSPACE_DIR, "intent_classifier.joblib")
TRAIN_DATASET_PATH = os.path.join(WORKSPACE_DIR, "data", "raw", "SCCU-test-enhanced.json")

def preprocess_query_simple(q):
    import re
    # Simple normalization for comparison
    return re.sub(r'\s+', ' ', q.strip().lower()) if isinstance(q, str) else q

def identify_lexical_gap(model_path: str, eval_data: list) -> dict:
    """
    Identifies the lexical gap between the model's vocabulary and the evaluation dataset.
    Returns a dictionary with:
      - 'eval_unique_words': set of all unique words found in evaluation data after preprocessing.
      - 'oov_words': set of evaluation words not present in the model's TF-IDF vocabulary.
      - 'oov_percentage': percentage of evaluation words that are OOV.
    """
    import re
    stemmer = PorterStemmer()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'which', 'who', 'when', 'where', 'why', 'how'
    }
    
    # 1. Load model and extract vocabulary
    try:
        model = joblib.load(model_path)
        tfidf = model.named_steps['tfidf']
        model_vocab = set(tfidf.vocabulary_.keys())
    except Exception as e:
        print(f"Warning: Could not load model for lexical gap identification: {e}")
        return None

    # 2. Extract words from evaluation queries
    eval_unique_words = set()
    oov_words = set()

    for item in eval_data:
        query = item.get("query", "")
        if not isinstance(query, str):
            continue
            
        # Preprocessing: lowercase, clean whitespace, remove stop words, and stem
        clean = query.lower().strip()
        clean = re.sub(r'\s+', ' ', clean)
        words = [w for w in clean.split() if w not in stop_words]
        stemmed_words = [stemmer.stem(w) for w in words]
        
        for w in stemmed_words:
            if w:
                eval_unique_words.add(w)
                if w not in model_vocab:
                    oov_words.add(w)

    oov_percentage = (len(oov_words) / len(eval_unique_words) * 100) if eval_unique_words else 0.0

    return {
        "eval_unique_words": eval_unique_words,
        "oov_words": oov_words,
        "oov_percentage": oov_percentage
    }

def main():
    print("=" * 60)
    print("EVALUATION DATASET DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # 1. Load JSON file
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"ERROR: Evaluation dataset not found at: {EVAL_DATASET_PATH}")
        return
        
    try:
        with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[SUCCESS] Loaded evaluation dataset. Total records: {len(data)}")
    except json.JSONDecodeError as jde:
        print(f"[CRITICAL ERROR] JSON is invalid/malformed: {jde}")
        return
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to read JSON: {e}")
        return

    if not isinstance(data, list):
        print("[CRITICAL ERROR] Root element is not a JSON list.")
        return

    # 2. Check structure and missing/empty fields
    malformed_records = []
    missing_query = []
    missing_actual = []
    empty_query = []
    empty_actual = []
    whitespace_query = []
    whitespace_actual = []
    valid_records = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            malformed_records.append((idx, f"Record is not a dictionary: {type(item)}"))
            continue
            
        has_query = "query" in item
        has_actual = "actual" in item
        
        # Check for unexpected/extra keys
        extra_keys = set(item.keys()) - {"query", "actual"}
        if extra_keys:
            malformed_records.append((idx, f"Unexpected keys: {extra_keys}"))
            
        if not has_query:
            missing_query.append(idx)
        if not has_actual:
            missing_actual.append(idx)
            
        q_val = item.get("query")
        a_val = item.get("actual")
        
        is_valid = True
        
        if has_query:
            if not isinstance(q_val, str):
                malformed_records.append((idx, f"Query is not a string: {type(q_val)}"))
                is_valid = False
            elif not q_val.strip():
                empty_query.append(idx)
                is_valid = False
            elif q_val != q_val.strip() or '  ' in q_val:
                whitespace_query.append((idx, q_val))
                
        if has_actual:
            if not isinstance(a_val, str):
                malformed_records.append((idx, f"Actual intent is not a string: {type(a_val)}"))
                is_valid = False
            elif not a_val.strip():
                empty_actual.append(idx)
                is_valid = False
            elif a_val != a_val.strip():
                whitespace_actual.append((idx, a_val))
                
        if is_valid and has_query and has_actual:
            valid_records.append(item)

    print("\n--- Structural & Formatting Checks ---")
    print(f"Malformed records (not dict, wrong types, extra keys): {len(malformed_records)}")
    for idx, err in malformed_records[:10]:
        print(f"  - Record #{idx}: {err}")
    if len(malformed_records) > 10:
        print(f"  ... and {len(malformed_records) - 10} more.")

    print(f"Missing 'query' key: {len(missing_query)}")
    if missing_query:
        print(f"  - Record indices: {missing_query[:10]}")
        
    print(f"Missing 'actual' key: {len(missing_actual)}")
    if missing_actual:
        print(f"  - Record indices: {missing_actual[:10]}")
        
    print(f"Empty/blank 'query': {len(empty_query)}")
    if empty_query:
        print(f"  - Record indices: {empty_query[:10]}")
        
    print(f"Empty/blank 'actual': {len(empty_actual)}")
    if empty_actual:
        print(f"  - Record indices: {empty_actual[:10]}")

    print(f"Queries with leading/trailing or multiple spaces: {len(whitespace_query)}")
    for idx, val in whitespace_query[:5]:
        print(f"  - Record #{idx}: {repr(val)}")
    if len(whitespace_query) > 5:
        print(f"  ... and {len(whitespace_query) - 5} more.")

    print(f"Actual intents with leading/trailing spaces: {len(whitespace_actual)}")
    for idx, val in whitespace_actual[:5]:
        print(f"  - Record #{idx}: {repr(val)}")

    # 3. Duplicate checks
    # We will normalize queries to lowercase & stripped for checking duplicates
    normalized_queries = [preprocess_query_simple(r['query']) for r in valid_records]
    query_counts = Counter(normalized_queries)
    
    duplicates = {q: count for q, count in query_counts.items() if count > 1}
    
    # Categorize duplicates:
    # A duplicate query can be "Redundant" (maps to same intent) or "Conflicting" (maps to different intents)
    redundant_duplicates = 0
    conflicting_duplicates = {}
    
    for dup_query in duplicates:
        # Find all records with this query
        matching_records = [r for r in valid_records if preprocess_query_simple(r['query']) == dup_query]
        intents = set(r['actual'] for r in matching_records)
        if len(intents) == 1:
            redundant_duplicates += 1
        else:
            conflicting_duplicates[dup_query] = list(intents)

    print("\n--- Duplicate & Consistency Checks ---")
    print(f"Total unique queries: {len(query_counts)}")
    print(f"Total queries with duplicates: {len(duplicates)}")
    print(f"  - Redundant duplicates (same intent mapped multiple times): {redundant_duplicates}")
    print(f"  - Conflicting duplicates (same query mapped to DIFFERENT intents): {len(conflicting_duplicates)}")
    for q, intents in list(conflicting_duplicates.items())[:10]:
        print(f"    * Query: \"{q}\" maps to: {intents}")
    if len(conflicting_duplicates) > 10:
        print(f"    ... and {len(conflicting_duplicates) - 10} more conflicting duplicate queries.")

    # 4. Valid intent classes checks
    # Let's get the list of valid intents either from the model classes or from the training dataset
    valid_intents = None
    source_name = ""
    
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            valid_intents = list(model.classes_)
            source_name = "trained model classes"
        except Exception:
            pass
            
    if not valid_intents and os.path.exists(TRAIN_DATASET_PATH):
        try:
            with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            utts = train_data['assets']['utterances']
            valid_intents = list(set(u['intent'] for u in utts))
            source_name = "training dataset (SCCU-test-enhanced.json)"
        except Exception as e:
            print(f"\nCould not load training data for comparison: {e}")
            
    if valid_intents:
        valid_intents_set = set(valid_intents)
        eval_intents = set(r['actual'] for r in valid_records)
        
        invalid_intents = eval_intents - valid_intents_set
        
        print(f"\n--- Intent Class Validation (Comparing to {source_name}) ---")
        print(f"Valid intents count in training/model: {len(valid_intents)}")
        print(f"Unique intents in evaluation dataset: {len(eval_intents)}")
        print(f"Evaluation intents NOT present in training/model: {len(invalid_intents)}")
        for intent in sorted(invalid_intents):
            # Check for possible case-insensitive matches
            ci_match = [vi for vi in valid_intents if vi.lower() == intent.lower()]
            if ci_match:
                print(f"  - \"{intent}\" (Typos/Casing? Did you mean: {ci_match})")
            else:
                print(f"  - \"{intent}\" (Completely unrecognized/missing in training)")
    else:
        print("\n--- Intent Class Validation ---")
        print("Warning: Could not load model or training dataset to check intent validity.")

    # 5. Lexical Gap Identification
    if os.path.exists(MODEL_PATH):
        gap_results = identify_lexical_gap(MODEL_PATH, valid_records)
        if gap_results:
            print("\n--- Lexical Gap Analysis (Vocabulary Coverage) ---")
            print(f"Unique preprocessed words in evaluation dataset: {len(gap_results['eval_unique_words'])}")
            print(f"Unique Out-of-Vocabulary (OOV) words in evaluation dataset: {len(gap_results['oov_words'])}")
            print(f"Lexical Gap (OOV rate): {gap_results['oov_percentage']:.2f}%")
            print("\nSample OOV Words (words evaluation dataset uses but training/model does not know):")
            sample_oov = sorted(list(gap_results['oov_words']))
            for w in sample_oov[:30]:
                print(f"  - {w}")
            if len(sample_oov) > 30:
                print(f"  ... and {len(sample_oov) - 30} more words.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
