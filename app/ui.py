import streamlit as st

def inject_custom_css():
    """Injects custom premium CSS styling into the Streamlit app."""
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

def render_header():
    """Renders the main page header and subtitle."""
    st.markdown("<h1 class='title-text'>🤖 Global Search Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Enter a customer query to automatically classify their intent using a pre-trained TF-IDF & Logistic Regression model.</p>", unsafe_allow_html=True)

def render_section_header(title: str):
    """Renders a custom section header with predefined styling."""
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

def render_result_card(query_text: str, preprocessed_tokens: str, prediction: str):
    """Renders a styled card showing the query prediction result."""
    tokens_display = preprocessed_tokens if preprocessed_tokens else '[Empty after stopword removal]'
    st.markdown(f"""
    <div class="card">
        <h3 style="margin-top:0; color:#1e293b;">Prediction Result</h3>
        <p style="margin-bottom: 8px;"><strong>Input Query:</strong> "{query_text}"</p>
        <p style="margin-bottom: 20px;"><strong>Preprocessed Tokens:</strong> <span class="preprocessed-badge">{tokens_display}</span></p>
        <div style="margin-bottom: 10px;">
            <span style="font-weight: 600; color: #475569; margin-right: 10px;">Predicted Intent:</span>
            <span class="intent-badge">{prediction}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_confidence_row(intent_name: str, prob_val: float, prediction: str):
    """Renders a confidence probability row and progress bar."""
    prob_pct = f"{prob_val * 100:.1f}%"
    is_predicted = (intent_name == prediction)
    color_style = "font-weight: 700; color: #10B981;" if is_predicted else "color: #1e293b;"
    bg_style = "background: linear-gradient(90deg, #10B981 0%, #059669 100%);" if is_predicted else ""
    
    st.markdown(f"""
    <div class="confidence-row">
        <span style="{color_style}">{intent_name}</span>
        <span style="font-weight: 600; color: #475569;">{prob_pct}</span>
    </div>
    <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width: {prob_val * 100}%; {bg_style}"></div>
    </div>
    """, unsafe_allow_html=True)
