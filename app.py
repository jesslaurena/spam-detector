"""
Spam Detector Web App using Streamlit
Compares Naive Bayes and KNN algorithms with timing information
"""

import streamlit as st
import sys
import os
import json
import time
from pathlib import Path

# Add NaiveBayes to path (both directory and src subdirectory)
naive_bayes_dir = Path(__file__).parent / "NaiveBayes"
if str(naive_bayes_dir) not in sys.path:
    sys.path.insert(0, str(naive_bayes_dir))

# Page config
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .algorithm-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .spam-result {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .ham-result {
        background-color: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛡️ Spam Detector</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.2rem; color: #6b7280;'>"
    "Email spam detection using Naive Bayes and KNN algorithms</p>",
    unsafe_allow_html=True
)

# Sidebar with info
with st.sidebar:
    st.header("📊 About")
    st.write("""
    This app compares two machine learning algorithms:
    
    - **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
    - **KNN**: K-Nearest Neighbors algorithm
    
    Both algorithms analyze email content to classify messages as **Spam** or **Ham** (legitimate).
    """)
    
    show_details = True
    
    st.header("ℹ️ How to Use")
    st.write("""
    1. Paste your email text in the text area below
    2. Click "Analyze Email"
    3. View results from both algorithms
    4. Compare performance metrics
    """)


def load_naive_bayes_model():
    """Load Naive Bayes model and vocabulary"""
    try:
        # Add NaiveBayes directory to Python path
        import sys
        naive_bayes_dir = Path(__file__).parent / "NaiveBayes"
        if str(naive_bayes_dir) not in sys.path:
            sys.path.insert(0, str(naive_bayes_dir))
        
        from src.model import MultinomialNB
        from src.vectorize import load_vocab, vectorize
        
        model_path = Path(__file__).parent / "NaiveBayes" / "models" / "nb_model.pkl"
        vocab_path = Path(__file__).parent / "NaiveBayes" / "models" / "vocab.json"
        
        if not model_path.exists():
            return None, None, "Model file not found. Please train the model first."
        if not vocab_path.exists():
            return None, None, "Vocabulary file not found. Please train the model first."
        
        model = MultinomialNB.load(str(model_path))
        vocab = load_vocab(str(vocab_path))
        
        return model, vocab, None
    except Exception as e:
        return None, None, f"Error loading Naive Bayes: {str(e)}"


def load_knn_model():
    """Load KNN model if available"""
    try:
        # Check if KNN directory exists
        knn_path = Path(__file__).parent / "KNN" / "models" / "knn_model.pkl"
        if not knn_path.exists():
            return None, None, "KNN model not yet implemented"
        
        # Add KNN to path if it exists
        sys.path.insert(0, str(Path(__file__).parent / "KNN" / "src"))
        
        # TODO: Implement KNN loading when ready

        return None, None, "KNN loading not yet implemented"
    except Exception as e:
        return None, None, f"Error loading KNN: {str(e)}"


def preprocess_text(text: str, lowercase: bool = True):
    """Simple text preprocessing"""
    import re
    if text is None:
        return []
    t = str(text)
    if lowercase:
        t = t.lower()
    return re.findall(r"[a-z0-9]+", t)


def predict_naive_bayes(text: str, model, vocab):
    """Predict using Naive Bayes"""
    # Add NaiveBayes directory to Python path if needed
    import sys
    naive_bayes_dir = Path(__file__).parent / "NaiveBayes"
    if str(naive_bayes_dir) not in sys.path:
        sys.path.insert(0, str(naive_bayes_dir))
    
    from src.vectorize import vectorize
    
    tokens = preprocess_text(text)
    x = vectorize(tokens, vocab, ngram_range=(1, 1), lowercase=True)
    
    label = model.predict(x)
    probs = model.predict_proba(x)
    
    # Convert label to int (0 or 1)
    # Model might return class name or number
    if isinstance(label, str):
        label = 1 if label.lower() in ['spam', '1'] else 0
    else:
        label = int(label)
    
    # Convert probs dict to have string keys for consistency
    # probs comes as {class: probability} where class might be 0, 1 or "0", "1"
    prob_dict = {}
    for key, val in probs.items():
        prob_dict[str(key)] = float(val)
    
    # Ensure both 0 and 1 keys exist
    if "0" not in prob_dict:
        prob_dict["0"] = 0.0
    if "1" not in prob_dict:
        prob_dict["1"] = 0.0
    
    return {
        "label": label,
        "probs": prob_dict
    }


def predict_knn(text: str, knn_model):
    """Predict using KNN - to be implemented"""

    # TODO: Implement KNN prediction when ready

    return None


# Load models at startup (cached)
@st.cache_resource
def load_models():
    """Load both models"""
    nb_model, nb_vocab, nb_error = load_naive_bayes_model()
    knn_model, knn_vocab, knn_error = load_knn_model()
    
    return {
        "naive_bayes": {"model": nb_model, "vocab": nb_vocab, "error": nb_error},
        "knn": {"model": knn_model, "vocab": knn_vocab, "error": knn_error}
    }


# Load models
models = load_models()

# Check if models are loaded
if models["naive_bayes"]["error"]:
    st.error(f"❌ Naive Bayes Error: {models['naive_bayes']['error']}")
    st.stop()

# Main input area
st.markdown("---")
st.subheader("📧 Enter Email Text")

email_text = st.text_area(
    "Paste the email content here:",
    height=200,
    placeholder="Enter the email text you want to analyze...",
    help="Paste any email text to check if it's spam or ham"
)

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🔍 Analyze Email",
        type="primary",
        use_container_width=True,
        disabled=not email_text.strip()
    )

# Run analysis
if analyze_button and email_text.strip():
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    results = {
        "naive_bayes": None,
        "knn": None,
        "times": {}
    }
    
    # Naive Bayes Prediction
    with st.spinner("Running Naive Bayes algorithm..."):
        try:
            nb_start = time.time()
            nb_result = predict_naive_bayes(
                email_text,
                models["naive_bayes"]["model"],
                models["naive_bayes"]["vocab"]
            )
            nb_time = (time.time() - nb_start) * 1000  # Convert to milliseconds
            nb_result["timeMs"] = nb_time
            results["naive_bayes"] = nb_result
            results["times"]["naive_bayes"] = nb_time
        except Exception as e:
            st.error(f"Naive Bayes prediction failed: {str(e)}")
    
    # KNN Prediction
    if models["knn"]["model"] is not None:
        with st.spinner("Running KNN algorithm..."):
            try:
                knn_start = time.time()
                knn_result = predict_knn(email_text, models["knn"]["model"])
                knn_time = (time.time() - knn_start) * 1000
                if knn_result:
                    knn_result["timeMs"] = knn_time
                    results["knn"] = knn_result
                    results["times"]["knn"] = knn_time
            except Exception as e:
                st.warning(f"KNN prediction failed: {str(e)}")
    
    # Display Results
    if results["naive_bayes"]:
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        # Naive Bayes Result
        with col1:
            st.markdown("### 🎯 Naive Bayes")
            nb_result = results["naive_bayes"]
            
            # Prediction result
            if nb_result["label"] == 1:
                st.markdown(
                    f'<div class="spam-result">🚨 SPAM</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ham-result">✅ HAM (Legitimate)</div>',
                    unsafe_allow_html=True
                )
            
            # Metrics
            label_key = str(nb_result['label'])
            confidence = nb_result['probs'].get(label_key, 0) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
            st.metric("Execution Time", f"{nb_result['timeMs']:.2f} ms")
            
            # Detailed probabilities
            if show_details:
                with st.expander("View Probabilities"):
                    st.write(f"**Ham (Legitimate):** {nb_result['probs'].get('0', 0) * 100:.2f}%")
                    st.write(f"**Spam:** {nb_result['probs'].get('1', 0) * 100:.2f}%")
        
        # KNN Result
        with col2:
            if results["knn"]:
                st.markdown("### 🔍 KNN")
                knn_result = results["knn"]
                
                # Prediction result
                if knn_result["label"] == 1:
                    st.markdown(
                        f'<div class="spam-result">🚨 SPAM</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="ham-result">✅ HAM (Legitimate)</div>',
                        unsafe_allow_html=True
                    )
                
                # Metrics
                label_key = str(knn_result['label'])
                confidence = knn_result['probs'].get(label_key, 0) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
                st.metric("Execution Time", f"{knn_result['timeMs']:.2f} ms")
                
                # Detailed probabilities
                if show_details:
                    with st.expander("View Probabilities"):
                        st.write(f"**Ham (Legitimate):** {knn_result['probs'].get('0', 0) * 100:.2f}%")
                        st.write(f"**Spam:** {knn_result['probs'].get('1', 0) * 100:.2f}%")
            else:
                st.markdown("### 🔍 KNN")
                st.info("KNN algorithm not yet implemented")
                st.markdown(
                    '<div style="text-align: center; padding: 2rem; color: #6b7280;">Coming soon...</div>',
                    unsafe_allow_html=True
                )
        
        # Comparison Section
        if results["knn"]:
            st.markdown("---")
            st.subheader("⚡ Performance Comparison")
            
            nb_time = results["times"]["naive_bayes"]
            knn_time = results["times"]["knn"]
            time_diff = abs(knn_time - nb_time)
            faster = "Naive Bayes" if nb_time < knn_time else "KNN"
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric("Faster Algorithm", faster)
            
            with comp_col2:
                st.metric("Time Difference", f"{time_diff:.2f} ms")
            
            with comp_col3:
                # Agreement check
                if nb_result["label"] == knn_result["label"]:
                    st.success("✅ Both algorithms agree")
                else:
                    st.warning("⚠️ Algorithms disagree")
            
            # Visual comparison
            st.markdown("#### Execution Time Comparison")
            import pandas as pd
            comparison_df = pd.DataFrame({
                "Algorithm": ["Naive Bayes", "KNN"],
                "Time (ms)": [nb_time, knn_time]
            })
            st.bar_chart(comparison_df.set_index("Algorithm"))

# Footer
st.markdown("---")

team_members = [
    {"name": "Jess Anderson", "linkedin": "https://www.linkedin.com/in/jessanderson1145/"},
    {"name": "Suchir Kolli", "linkedin": "https://www.linkedin.com/in/suchir-kolli-9a5288293/"},
    {"name": "Isabella Pareja", "linkedin": "https://www.linkedin.com/in/isabella-pareja-407484380/"},
]

links_html = " • ".join([
    f'<a href="{member["linkedin"]}" target="_blank" style="color: #1f77b4; text-decoration: none; font-weight: 500;">{member["name"]}</a>'
    for member in team_members
])

st.markdown(
    f"<div style='text-align: center; color: #6b7280; margin-top: 1rem;'>Built by {links_html}</div>",
    unsafe_allow_html=True
)

