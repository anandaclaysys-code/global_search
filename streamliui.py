import streamlit as st
import joblib
import re
import os
import json
import pandas as pd
from nltk.stem import PorterStemmer

# Set page config
st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Title styling */
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #475569;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Intent result badge */
    .intent-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    
    /* Preprocessed text badge */
    .preprocessed-badge {
        background-color: #f1f5f9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
        border: 1px solid #cbd5e1;
    }
    
    /* Section headers */
    .section-header {
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 0.3rem;
    }
    
    /* Confidence row */
    .confidence-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .confidence-bar-bg {
        background-color: #e2e8f0;
        border-radius: 10px;
        height: 10px;
        width: 100%;
        margin-bottom: 15px;
        overflow: hidden;
    }
    
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
    }
</style>
""", unsafe_allow_html=True)

# Stemmer and stop words matching the training phase
@st.cache_resource
def load_stemmer_and_stopwords():
    stemmer = PorterStemmer()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'which', 'who', 'when', 'where', 'why', 'how'
    }
    return stemmer, stop_words

stemmer, stop_words = load_stemmer_and_stopwords()

def preprocess(query: str) -> str:
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
def load_model(model_path):
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

model_path = 'intent_classifier.joblib'
model = load_model(model_path)

# Load dataset metadata if exists
@st.cache_data
def get_dataset_info(json_path):
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

dataset_path = 'SCCU-test.json'
num_samples, num_intents, all_intents = get_dataset_info(dataset_path)

# UI Elements
st.markdown("<h1 class='title-text'>🤖 Digital Banking Intent Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Enter a customer query to automatically classify their intent using a pre-trained TF-IDF & Logistic Regression model.</p>", unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.markdown("### ⚙️ System Status & Config")
    if model is not None:
        st.success("Model loaded successfully!")
        # Inspect model pipeline
        try:
            tfidf_step = model.named_steps['tfidf']
            clf_step = model.named_steps['clf']
            st.info(f"**Vectorizer**: TF-IDF (N-grams {tfidf_step.ngram_range})")
            st.info(f"**Classifier**: {clf_step.__class__.__name__} (C={clf_step.C})")
        except Exception:
            st.info("Pipeline: Custom / Scikit-learn Pipeline")
    else:
        st.error("Model file `intent_classifier.joblib` not found. Please train it first.")
        
    if num_samples is not None:
        st.markdown("### 📊 Dataset Properties")
        st.write(f"- **Total Training Utterances**: {num_samples}")
        st.write(f"- **Unique Intent Categories**: {num_intents}")
        with st.expander("Show All Intent Categories"):
            st.write(", ".join(sorted(all_intents)))

# Initialize session state for text input if not exists
if 'query_text_input' not in st.session_state:
    st.session_state.query_text_input = ""

# Callback for example buttons
def set_query(text):
    st.session_state.query_text_input = text

st.markdown("<div class='section-header'>💡 Try an Example Query</div>", unsafe_allow_html=True)
examples = [
    ("Transfer $500 from savings to checking", "Transfer money"),
    ("What is my account balance?", "Accounts info"),
    ("Apply for a new auto loan", "Auto loans"),
    ("Lock my debit card", "Card controls"),
    ("Pay my monthly electric bill", "Bill pay")
]

# Display buttons in columns
cols = st.columns(len(examples))
for idx, (label, desc) in enumerate(examples):
    with cols[idx]:
        st.button(label, key=f"ex_{idx}", help=desc, use_container_width=True, on_click=set_query, args=(label,))

# Main Query input area
st.markdown("<div class='section-header'>🔍 Run Intent Prediction</div>", unsafe_allow_html=True)
query_text = st.text_input(
    "Enter customer utterance:",
    value=st.session_state.query_text_input,
    placeholder="Type a request (e.g. 'Can I deposit a check using my phone?')",
    key="query_text_input"
)

# Run classification if we have a query
if query_text:
    if model is None:
        st.warning("Please make sure `intent_classifier.joblib` is generated by running the training script.")
    else:
        # Preprocess
        preprocessed = preprocess(query_text)
        
        # Predict
        prediction = model.predict([preprocessed])[0]
        
        # Calculate probabilities if available
        try:
            probabilities = model.predict_proba([preprocessed])[0]
            classes = model.classes_
            prob_df = pd.DataFrame({
                'intent': classes,
                'probability': probabilities
            }).sort_values(by='probability', ascending=False).reset_index(drop=True)
        except Exception:
            prob_df = None

        # Display result card
        st.markdown(f"""
        <div class="card">
            <h3 style="margin-top:0; color:#1e293b;">Prediction Result</h3>
            <p style="margin-bottom: 8px;"><strong>Input Query:</strong> "{query_text}"</p>
            <p style="margin-bottom: 20px;"><strong>Preprocessed Tokens:</strong> <span class="preprocessed-badge">{preprocessed if preprocessed else '[Empty after stopword removal]'}</span></p>
            <div style="margin-bottom: 10px;">
                <span style="font-weight: 600; color: #475569; margin-right: 10px;">Predicted Intent:</span>
                <span class="intent-badge">{prediction}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show top probabilities chart/bars
        if prob_df is not None:
            st.markdown("<div class='section-header'>📈 Confidence Distribution (Top 5)</div>", unsafe_allow_html=True)
            top_5 = prob_df.head(5)
            
            for idx, row in top_5.iterrows():
                intent_name = row['intent']
                prob_val = row['probability']
                prob_pct = f"{prob_val * 100:.1f}%"
                
                # Highlight the predicted one
                color_style = "font-weight: 700; color: #10B981;" if intent_name == prediction else "color: #1e293b;"
                
                st.markdown(f"""
                <div class="confidence-row">
                    <span style="{color_style}">{intent_name}</span>
                    <span style="font-weight: 600; color: #475569;">{prob_pct}</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: {prob_val * 100}%; {'background: linear-gradient(90deg, #10B981 0%, #059669 100%);' if intent_name == prediction else ''}"></div>
                </div>
                """, unsafe_allow_html=True)
