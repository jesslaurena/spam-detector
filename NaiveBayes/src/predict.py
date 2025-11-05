"""CLI to predict spam/ham with TF‑IDF classifier (preferred) or Naive Bayes (fallback)."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

# Ensure repo root is importable for `from src...`
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.vectorize import load_vocab, vectorize
from src.model import MultinomialNB

# joblib used for TF-IDF artifacts (optional)
try:
    import joblib
except Exception:
    joblib = None


def _fallback_preprocess(text: str, lowercase: bool = True) -> List[str]:
    """Simple tokenizer used if src.preprocess.preprocess is not available."""
    import re
    if text is None:
        return []
    t = str(text)
    if lowercase:
        t = t.lower()
    return re.findall(r"[a-z0-9]+", t)


def load_nb_artifacts(model_path: str, vocab_path: str) -> Tuple[MultinomialNB, Dict[str, int]]:
    """Load NB model and vocab from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    model = MultinomialNB.load(model_path)
    vocab = load_vocab(vocab_path)
    return model, vocab


def predict_with_nb_text(
    text: str,
    model: MultinomialNB,
    vocab: Dict[str, int],
    preprocess_fn,
    ngram_range=(1, 1),
    lowercase: bool = True,
) -> Dict[str, Any]:
    tokens = preprocess_fn(text, lowercase) if preprocess_fn is not None else _fallback_preprocess(text, lowercase)
    x = vectorize(tokens, vocab, ngram_range=tuple(ngram_range), lowercase=lowercase)
    label = model.predict(x)
    probs = model.predict_proba(x)
    return {"text": text, "label": label, "probs": probs}


def predict_with_nb_texts(
    texts: Iterable[str],
    model: MultinomialNB,
    vocab: Dict[str, int],
    preprocess_fn,
    ngram_range=(1, 1),
    lowercase: bool = True,
) -> List[Dict[str, Any]]:
    return [predict_with_nb_text(t, model, vocab, preprocess_fn, ngram_range=ngram_range, lowercase=lowercase) for t in texts]


def predict_with_tfidf_text(
    text: str,
    vect,
    clf,
    preprocess_fn,
    lowercase: bool = True,
) -> Dict[str, Any]:
    """Predict a single text using TF-IDF vectorizer + classifier."""
    if preprocess_fn is not None:
        tokens = preprocess_fn(text, lowercase)
        text_for_vect = " ".join(tokens)
    else:
        text_for_vect = text
    X = vect.transform([text_for_vect])
    prob_arr = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]
    return {
        "text": text,
        "label": int(pred) if hasattr(pred, "__int__") else str(pred),
        "probs": {str(i): float(p) for i, p in enumerate(prob_arr)}
    }


def predict_with_tfidf_texts(
    texts: Iterable[str],
    vect,
    clf,
    preprocess_fn,
    lowercase: bool = True,
) -> List[Dict[str, Any]]:
    if preprocess_fn is not None:
        prepared = [" ".join(preprocess_fn(t, lowercase)) for t in texts]
    else:
        prepared = list(texts)
    X = vect.transform(prepared)
    prob_matrix = clf.predict_proba(X)
    preds = clf.predict(X)
    results: List[Dict[str, Any]] = []
    for orig_text, pred, probs in zip(texts, preds, prob_matrix):
        results.append({
            "text": orig_text,
            "label": int(pred) if hasattr(pred, "__int__") else str(pred),
            "probs": {str(i): float(p) for i, p in enumerate(probs)}
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict with trained classifier (TF-IDF preferred, NB fallback)")
    parser.add_argument("--model", default="models/nb_model.pkl", help="Path to saved NB model (pickle)")
    parser.add_argument("--vocab", default="models/vocab.json", help="Path to saved NB vocab (JSON)")
    parser.add_argument("--tfidf-model", default="NaiveBayes/models/logreg_tfidf.pkl", help="Path to TF-IDF classifier (joblib)")
    parser.add_argument("--tfidf-vect", default="NaiveBayes/models/tfidf_vectorizer.pkl", help="Path to TF-IDF vectorizer (joblib)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text to classify (wrap in quotes)")
    group.add_argument("--input-file", help='Path to input file (one text per line or JSONL with {"text":...})')
    parser.add_argument("--output-file", help="Path to save predictions as JSON or JSONL (if not provided prints to stdout)")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=[1, 1], help="ngram range min max (for NB fallback)")
    parser.add_argument("--lowercase", action="store_true", help="lowercase texts before tokenization")
    args = parser.parse_args()

    ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))
    lowercase = bool(args.lowercase)

    # Try TF-IDF artifacts first
    tfidf_available = False
    vect = clf = None
    tfidf_vect_path = Path(args.tfidf_vect)
    tfidf_model_path = Path(args.tfidf_model)
    if joblib is not None and tfidf_vect_path.exists() and tfidf_model_path.exists():
        try:
            vect = joblib.load(str(tfidf_vect_path))
            clf = joblib.load(str(tfidf_model_path))
            tfidf_available = True
        except Exception:
            tfidf_available = False

    # Load NB artifacts as fallback
    nb_model = nb_vocab = None
    nb_available = False
    if not tfidf_available:
        try:
            nb_model, nb_vocab = load_nb_artifacts(args.model, args.vocab)
            nb_available = True
        except Exception as e:
            raise SystemExit(f"Failed to load NB artifacts and TF-IDF not available: {e}")

    # Try to import optional team preprocess
    try:
        from src.preprocess import preprocess  # type: ignore
        preprocess_fn = preprocess
    except Exception:
        preprocess_fn = None

    # Prepare inputs
    texts: List[str] = []
    if args.text:
        texts = [args.text]
    else:
        if not os.path.exists(args.input_file):
            raise SystemExit(f"Input file not found: {args.input_file}")
        with open(args.input_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "text" in obj:
                            texts.append(str(obj["text"]))
                            continue
                    except Exception:
                        pass
                texts.append(line)

    # Run predictions
    if tfidf_available:
        results = predict_with_tfidf_texts(texts, vect, clf, preprocess_fn, lowercase=lowercase)
    else:
        results = predict_with_nb_texts(texts, nb_model, nb_vocab, preprocess_fn or _fallback_preprocess, ngram_range=ngram_range, lowercase=lowercase)

    # Output results
    if args.output_file:
        out_path = args.output_file
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if len(results) == 1:
            with open(out_path, "w", encoding="utf8") as f:
                json.dump(results[0], f, indent=2)
        else:
            with open(out_path, "w", encoding="utf8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} prediction(s) to {out_path}")
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()