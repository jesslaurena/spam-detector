# app.py
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

# ----- Helpers to load models -----
def load_naive_bayes_model():
    """Load Naive Bayes model and vocabulary (original NB pipeline)."""
    try:
        import sys
        naive_bayes_dir = Path(__file__).parent / "NaiveBayes"
        if str(naive_bayes_dir) not in sys.path:
            sys.path.insert(0, str(naive_bayes_dir))
        
        from src.model import MultinomialNB
        from src.vectorize import load_vocab
        
        # Prefer NB model trained on raw text if available; fall back to BoW-trained NB if present.
        model_path = Path(__file__).parent / "NaiveBayes" / "models" / "nb_model.pkl"
        if not model_path.exists():
            # try nb_model_from_bow.pkl
            alt = Path(__file__).parent / "NaiveBayes" / "models" / "nb_model_from_bow.pkl"
            if alt.exists():
                model_path = alt

        vocab_path = Path(__file__).parent / "NaiveBayes" / "models" / "vocab.json"
        # if vocab doesn't exist, NB pipeline may be trained from dense BoW features; that's OK, return model with no vocab
        model = None
        vocab = None
        if model_path.exists():
            model = MultinomialNB.load(str(model_path))
        else:
            return None, None, "Model file not found. Please train the model first."
        if vocab_path.exists():
            vocab = load_vocab(str(vocab_path))
        # return model and vocab (vocab may be None when model trained from BoW dense features)
        return model, vocab, None
    except Exception as e:
        return None, None, f"Error loading Naive Bayes: {str(e)}"

def load_tfidf_artifacts():
    """Try to load TF-IDF vectorizer + classifier (joblib)."""
    try:
        import joblib
        tfidf_vect_path = Path(__file__).parent / "NaiveBayes" / "models" / "tfidf_vectorizer.pkl"
        tfidf_clf_path  = Path(__file__).parent / "NaiveBayes" / "models" / "logreg_tfidf.pkl"
        if tfidf_vect_path.exists() and tfidf_clf_path.exists():
            vect = joblib.load(str(tfidf_vect_path))
            clf = joblib.load(str(tfidf_clf_path))
            return vect, clf, None
        else:
            return None, None, "TF-IDF artifacts not found"
    except Exception as e:
        return None, None, f"Error loading TF-IDF artifacts: {str(e)}"

def load_knn_model():
    """
    Import KNN module at KNN/src/knn_model.py, load training features, build normalized index.
    Returns knn_obj dict with keys: module, X_train (np.array), X_train_norm (np.array), y_train (np.array)
    """
    try:
        import sys, numpy as np
        from pathlib import Path

        knn_src_dir = Path(__file__).parent / "KNN" / "src"
        if str(knn_src_dir) not in sys.path:
            sys.path.insert(0, str(knn_src_dir))

        import knn_model  # KNN/src/knn_model.py

        # expected CSV (training TF-IDF features)
        train_path = Path(__file__).parent.parent / "data" / "train_tfidf_features.csv"
        if not train_path.exists():
            # fallback to local data folder
            train_path = Path(__file__).parent / ".." / "data" / "train_tfidf_features.csv"
            train_path = train_path.resolve()
        if not train_path.exists():
            return None, None, f"Training features not found at expected path: {train_path}"

        # load_features should return (features_list, labels_list)
        train_features, train_labels = knn_model.load_features(str(train_path))
        X_train = np.asarray(train_features, dtype=np.float64)
        y_train = np.asarray(train_labels, dtype=np.int32)

        # If knn_model provides build_knn_index, use it; otherwise build normalized rows here
        if hasattr(knn_model, "build_knn_index"):
            X_train_norm = knn_model.build_knn_index(X_train)
        else:
            norms = np.linalg.norm(X_train, axis=1)
            nonzero = norms > 0
            X_train_norm = np.zeros_like(X_train, dtype=np.float64)
            if nonzero.any():
                X_train_norm[nonzero] = X_train[nonzero] / norms[nonzero, None]

        knn_obj = {
            "module": knn_model,
            "X_train": X_train,
            "X_train_norm": X_train_norm,
            "y_train": y_train
        }
        return knn_obj, None, None
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

# Robust helper to normalize predict_proba output to dict of string keys
def normalize_proba_output(raw_probs, classes=None):
    """
    raw_probs can be:
     - dict-like {class: prob}
     - numpy array (ordered according to classes param)
    Return dict of string->float mapping, keys are str(class)
    """
    try:
        # dict-like
        if isinstance(raw_probs, dict):
            return {str(k): float(v) for k, v in raw_probs.items()}
        # if it's an array and classes provided
        import numpy as _np
        if hasattr(raw_probs, "__iter__") and classes is not None:
            return {str(c): float(p) for c, p in zip(classes, list(raw_probs))}
        # fallback: try to coerce to list
        vals = list(raw_probs)
        if classes is not None:
            return {str(c): float(p) for c, p in zip(classes, vals)}
        # else return indexed keys
        return {str(i): float(p) for i, p in enumerate(vals)}
    except Exception:
        return {}

# Prediction wrappers
def predict_naive_bayes(text: str, model, vocab, tfidf_tuple=None):
    """
    Prefer to use NB model + vocab when vocab exists and is consistent.
    If NB vocab is None (model trained from dense BoW) or if TF-IDF artifacts are provided and NB vocab missing,
    fall back to TF-IDF classifier for prediction.
    tfidf_tuple: (vect, clf) or None
    """
    # If model missing, return error
    if model is None:
        return {"label": None, "probs": {"0": 0.0, "1": 0.0}, "error": "NB model not loaded"}

    # If no vocab (e.g., NB trained from dense BoW) and TF-IDF available, use TF-IDF classifier instead
    if (vocab is None or len(vocab) == 0) and tfidf_tuple is not None:
        vect, clf = tfidf_tuple
        # Try to preprocess similarly to TF-IDF training; we have a simple fallback
        try:
            from src.preprocess import preprocess as team_preprocess  # type: ignore
            tokens = team_preprocess(text, True)
            text_for_vect = " ".join(tokens)
        except Exception:
            text_for_vect = text
        X = vect.transform([text_for_vect])
        pred = clf.predict(X)[0]
        probs_raw = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else []
        proba = normalize_proba_output(probs_raw, classes=getattr(clf, "classes_", None))
        # ensure keys '0' and '1' exist
        if "0" not in proba:
            proba.setdefault("0", 0.0)
        if "1" not in proba:
            proba.setdefault("1", 0.0)
        # normalize label to int if possible
        try:
            label = int(pred)
        except Exception:
            label = 1 if str(pred).lower() in ("1", "spam", "true") else 0
        return {"label": label, "probs": proba}

    # Otherwise use NB model + vocab path
    try:
        from src.vectorize import vectorize
        tokens = preprocess_text(text, lowercase=True)
        x = vectorize(tokens, vocab, ngram_range=(1,1), lowercase=True) if vocab is not None else {}
        label_raw = model.predict(x)
        probs_raw = model.predict_proba(x)
        proba = normalize_proba_output(probs_raw, classes=getattr(model, "classes_", None))
        # Ensure both keys '0' and '1' exist
        proba.setdefault("0", 0.0)
        proba.setdefault("1", 0.0)
        # normalize label
        if isinstance(label_raw, str):
            label = 1 if label_raw.lower() in ['spam', '1'] else 0
        else:
            try:
                label = int(label_raw)
            except Exception:
                label = 1 if str(label_raw).lower() in ("1", "spam", "true") else 0
        return {"label": label, "probs": proba}
    except Exception as e:
        return {"label": None, "probs": {"0": 0.0, "1": 0.0}, "error": str(e)}

def predict_knn(text: str, knn_obj, tfidf_tuple=None, k=9):
    """
    Use the knn_obj loaded above and the TF-IDF vectorizer to predict.
    - knn_obj: dict from load_knn_model
    - tfidf_tuple: (vect, clf) from TF-IDF artifacts (we use vect to transform raw text into features)
    - k: neighbors
    Returns dict {"label": int, "probs": {"0": float, "1": float}} or an error field.
    """
    if knn_obj is None:
        return None

    try:
        # need TF-IDF vectorizer to convert raw text into same numeric features as training CSV
        if tfidf_tuple is None or tfidf_tuple[0] is None:
            return {"label": None, "probs": {"0": 0.0, "1": 0.0}, "error": "TF-IDF vectorizer required for KNN"}

        vect = tfidf_tuple[0]
        knn_mod = knn_obj["module"]
        X_train = knn_obj["X_train"]
        X_train_norm = knn_obj["X_train_norm"]
        y_train = knn_obj["y_train"]

        # preprocess -> build input string for vect
        try:
            from src.preprocess import preprocess as team_preprocess  # type: ignore
            tokens = team_preprocess(text, True)
            text_for_vect = " ".join(tokens)
        except Exception:
            text_for_vect = text

        X = vect.transform([text_for_vect])
        # convert sparse -> dense if necessary (match load_features representation)
        if hasattr(X, "toarray"):
            dense = X.toarray()[0].astype(float).tolist()
        else:
            dense = list(map(float, X[0]))

        # Prefer module's knn_predict_single if available (from the from_scratch code)
        if hasattr(knn_mod, "knn_predict_single"):
            res = knn_mod.knn_predict_single(X_train_norm, y_train, dense, k)
            # ensure consistent formatting
            res["probs"].setdefault("0", 0.0); res["probs"].setdefault("1", 0.0)
            res["label"] = int(res["label"])
            return res

        # Otherwise fall back to original teammate functions (get_k_nearest + majority vote)
        if hasattr(knn_mod, "get_k_nearest"):
            neighbors = knn_mod.get_k_nearest(X_train.tolist(), y_train.tolist(), dense, k)
            one = sum(1 for v in neighbors if int(v) == 1)
            zero = k - one
            label = 1 if one >= zero else 0
            conf = max(one, zero) / float(k)
            proba = {"0": 0.0, "1": 0.0}
            proba[str(label)] = float(conf)
            return {"label": int(label), "probs": proba}

        return {"label": None, "probs": {"0": 0.0, "1": 0.0}, "error": "KNN module lacks expected functions"}
    except Exception as e:
        return {"label": None, "probs": {"0": 0.0, "1": 0.0}, "error": str(e)}

# Load models at startup (cached)
@st.cache_resource
def load_models():
    """Load NB and TF-IDF artifacts"""
    nb_model, nb_vocab, nb_error = load_naive_bayes_model()
    tfidf_vect, tfidf_clf, tfidf_error = load_tfidf_artifacts()
    knn_model, knn_vocab, knn_error = load_knn_model()
    return {
        "naive_bayes": {"model": nb_model, "vocab": nb_vocab, "error": nb_error},
        "tfidf": {"vect": tfidf_vect, "clf": tfidf_clf, "error": tfidf_error},
        "knn": {"model": knn_model, "vocab": knn_vocab, "error": knn_error}
    }

# Load models
models = load_models()

# If NB model missing but TF-IDF present, we will use TF-IDF as fallback for predictions
if models["naive_bayes"]["error"] and models["tfidf"]["error"]:
    st.error("❌ No usable model found. Please train the model first.")
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
    
    # Naive Bayes Prediction (use TF-IDF fallback if necessary)
    with st.spinner("Running Naive Bayes algorithm..."):
        try:
            nb_start = time.time()
            tfidf_tuple = None
            if models["tfidf"]["error"] is None:
                tfidf_tuple = (models["tfidf"]["vect"], models["tfidf"]["clf"])
            nb_result = predict_naive_bayes(
                email_text,
                models["naive_bayes"]["model"],
                models["naive_bayes"]["vocab"],
                tfidf_tuple=tfidf_tuple
            )
            nb_time = (time.time() - nb_start) * 1000  # ms
            nb_result["timeMs"] = nb_time
            results["naive_bayes"] = nb_result
            results["times"]["naive_bayes"] = nb_time
        except Exception as e:
            st.error(f"Naive Bayes prediction failed: {str(e)}")
    
    # KNN Prediction (unchanged)
    if models["knn"]["model"] is not None:
        with st.spinner("Running KNN algorithm..."):
            try:
                knn_start = time.time()
                # Build tfidf_tuple only if TF-IDF loaded successfully
                tfidf_tuple = None
                if models["tfidf"]["error"] is None:
                    tfidf_tuple = (models["tfidf"]["vect"], models["tfidf"]["clf"])
                # choose k (use the same k your teammate expects, e.g., 9)
                knn_k = 9
                knn_result = predict_knn(email_text, models["knn"]["model"], tfidf_tuple=tfidf_tuple, k=knn_k)
                knn_time = (time.time() - knn_start) * 1000
                if knn_result:
                    knn_result["timeMs"] = knn_time
                    results["knn"] = knn_result
                    results["times"]["knn"] = knn_time
            except Exception as e:
                st.warning(f"KNN prediction failed: {str(e)}")
    
    # Display Results (UI unchanged)
    if results["naive_bayes"]:
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        # Naive Bayes Result
        with col1:
            st.markdown("### 🎯 Naive Bayes")
            nb_result = results["naive_bayes"]
            
            # Prediction result
            if nb_result.get("label") == 1:
                st.markdown(
                    f'<div class="spam-result">🚨 SPAM</div>',
                    unsafe_allow_html=True
                )
            elif nb_result.get("label") == 0:
                st.markdown(
                    f'<div class="ham-result">✅ HAM (Legitimate)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ham-result">❓ Unknown</div>',
                    unsafe_allow_html=True
                )
            
            # Metrics
            label_key = str(nb_result.get('label', '1'))
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