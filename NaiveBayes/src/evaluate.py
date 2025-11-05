"""
Evaluate a saved classifier on a labeled CSV.

Supports:
- Naive Bayes (saved model + vocab)
- KNN
- TF-IDF classifier (joblib vectorizer + classifier)

The script exposes functions for programmatic use and a CLI (main).
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Iterable, Optional

import pandas as pd
from sklearn import metrics

# Make repo root importable so `from src...` works when this script is run directly.
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.model import MultinomialNB
from src.vectorize import vectorize, load_vocab

# optional joblib for TF-IDF artifacts
try:
    import joblib
except Exception:
    joblib = None


def _fallback_preprocess(text: str, lowercase: bool = True) -> List[str]:
    """Simple fallback tokenizer used when src.preprocess is unavailable."""
    import re
    if text is None:
        return []
    t = str(text)
    if lowercase:
        t = t.lower()
    return re.findall(r"[a-z0-9]+", t)


def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str], pos_label: Optional[str] = "spam") -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, f1, confusion matrix and classification report.
    Uses binary pos_label when applicable, otherwise macro averages.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))

    accuracy = float(metrics.accuracy_score(y_true, y_pred))

    try:
        if pos_label is not None and pos_label in labels and len(labels) == 2:
            precision = float(metrics.precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            recall = float(metrics.recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            f1 = float(metrics.f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        else:
            precision = float(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0))
            recall = float(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0))
            f1 = float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0))
    except Exception:
        precision = float(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0))
        recall = float(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0))

    conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist()
    class_report = metrics.classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
        "confusion_matrix": conf_mat,
        "classification_report": class_report
    }


def evaluate_model_on_df(
    df: pd.DataFrame,
    model: MultinomialNB,
    vocab: Dict[str, int],
    preprocess_fn=None,
    ngram_range: Tuple[int, int] = (1, 1),
    lowercase: bool = True,
    pos_label: Optional[str] = "spam"
) -> Dict[str, Any]:
    """
    Evaluate a saved Naive Bayes model on a DataFrame with 'text' and 'label' columns.
    Returns metrics and a list of predictions (text, true_label, pred_label, probs).
    """
    if 'text' not in df.columns or 'label' not in df.columns:
        # try a common alternative
        if 'message' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'message': 'text'})
        else:
            raise ValueError("DataFrame must contain 'text' and 'label' columns")

    preprocess = preprocess_fn if preprocess_fn is not None else _fallback_preprocess
    texts = df['text'].fillna('').astype(str).tolist()
    true_labels = df['label'].astype(str).tolist()

    preds: List[str] = []
    probs: List[Dict[str, float]] = []

    for t in texts:
        tokens = preprocess(t, lowercase)
        x = vectorize(tokens, vocab, ngram_range=ngram_range, lowercase=lowercase)
        pred = model.predict(x)
        pred_probs = model.predict_proba(x)
        preds.append(str(pred))
        probs.append({str(k): float(v) for k, v in pred_probs.items()})

    metrics_report = compute_metrics(true_labels, preds, pos_label=pos_label)
    result = {
        "metrics": metrics_report,
        "predictions": [
            {"text": txt, "true_label": true, "pred_label": pred, "probs": prob}
            for txt, true, pred, prob in zip(texts, true_labels, preds, probs)
        ]
    }
    return result


def evaluate_tfidf_on_df(
    df: pd.DataFrame,
    vect,
    clf,
    preprocess_fn=None,
    lowercase: bool = True,
    pos_label: Optional[str] = "1"
) -> Dict[str, Any]:
    """
    Evaluate a TF-IDF classifier. vect should be a fitted vectorizer, clf a fitted classifier.
    Returns metrics and predictions (text, true_label, pred_label, probs).
    """
    if 'text' not in df.columns or 'label' not in df.columns:
        if 'message' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'message': 'text'})
        else:
            raise ValueError("DataFrame must contain 'text' and 'label' columns")

    preprocess = preprocess_fn if preprocess_fn is not None else None
    texts_raw = df['text'].fillna('').astype(str).tolist()
    if preprocess is not None:
        texts_for_vect = [" ".join(preprocess(t, lowercase)) for t in texts_raw]
    else:
        texts_for_vect = texts_raw

    X = vect.transform(texts_for_vect)
    y_pred_arr = clf.predict(X)
    preds = [str(p) for p in y_pred_arr]
    true_labels = df['label'].astype(str).tolist()

    probs: List[Dict[str, float]] = []
    if hasattr(clf, "predict_proba"):
        prob_matrix = clf.predict_proba(X)
        classes = clf.classes_
        for row_probs in prob_matrix:
            probs.append({str(c): float(p) for c, p in zip(classes, row_probs)})
    else:
        probs = [{} for _ in preds]

    metrics_report = compute_metrics(true_labels, preds, pos_label=pos_label)
    result = {
        "metrics": metrics_report,
        "predictions": [
            {"text": txt, "true_label": true, "pred_label": pred, "probs": prob}
            for txt, true, pred, prob in zip(texts_raw, true_labels, preds, probs)
        ]
    }
    return result


def save_report(report: Dict[str, Any], path: str) -> None:
    """Save a JSON report to disk (creates directories if needed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV and normalize common alternate column names to 'text' and 'label'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)

    if 'text' not in df.columns:
        for cand in ['message', 'body', 'Text', 'content', 'email_content']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'text'})
                break

    if 'label' not in df.columns:
        for cand in ['spam', 'labelled', 'Label', 'target']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'label'})
                break

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")

    return df[['text', 'label']]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model (NB or TF-IDF) on a labeled CSV")
    parser.add_argument("--data", required=True, help="Path to CSV with columns 'text' and 'label'")
    parser.add_argument("--model", required=False, help="Path to saved NB model pickle (nb_model.pkl)")
    parser.add_argument("--vocab", required=False, help="Path to vocab JSON (vocab.json)")
    parser.add_argument("--tfidf-model", required=False, help="Path to TF-IDF classifier (joblib pickle)")
    parser.add_argument("--tfidf-vect", required=False, help="Path to TF-IDF vectorizer (joblib pickle)")
    parser.add_argument("--output", default=None, help="Path to save evaluation report JSON")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=[1, 1], help="ngram range min max (NB only)")
    parser.add_argument("--lowercase", action="store_true", help="lowercase texts before tokenization")
    parser.add_argument("--pos-label", default=None, help="positive label for binary metrics (e.g. 'spam' or '1')")
    args = parser.parse_args()

    # load dataset
    try:
        df = load_dataset(args.data)
    except Exception as e:
        raise SystemExit(f"Failed to load data: {e}")

    # attempt to import team preprocess
    try:
        from src.preprocess import preprocess  # type: ignore
        preprocess_fn = preprocess
    except Exception:
        preprocess_fn = None

    lowercase = bool(args.lowercase)
    ngram_range = tuple(args.ngram_range)

    # prefer TF-IDF if artifacts provided & joblib available
    tfidf_path_ok = False
    tfidf_vect = tfidf_clf = None
    if args.tfidf_model and args.tfidf_vect and joblib is not None:
        if Path(args.tfidf_model).exists() and Path(args.tfidf_vect).exists():
            try:
                tfidf_clf = joblib.load(args.tfidf_model)
                tfidf_vect = joblib.load(args.tfidf_vect)
                tfidf_path_ok = True
            except Exception as e:
                print(f"Warning: failed to load TF-IDF artifacts: {e}")
                tfidf_path_ok = False

    if tfidf_path_ok:
        pos_label = args.pos_label if args.pos_label is not None else "1"
        result = evaluate_tfidf_on_df(df, tfidf_vect, tfidf_clf, preprocess_fn=preprocess_fn, lowercase=lowercase, pos_label=pos_label)
    else:
        if not args.model or not args.vocab:
            raise SystemExit("NB model and vocab must be provided when TF-IDF artifacts are not available. Use --model and --vocab.")
        try:
            nb_model = MultinomialNB.load(args.model)
        except Exception as e:
            raise SystemExit(f"Failed to load NB model: {e}")
        try:
            nb_vocab = load_vocab(args.vocab)
        except Exception as e:
            raise SystemExit(f"Failed to load vocab: {e}")
        pos_label = args.pos_label if args.pos_label is not None else "spam"
        result = evaluate_model_on_df(
            df,
            nb_model,
            nb_vocab,
            preprocess_fn=preprocess_fn,
            ngram_range=ngram_range,
            lowercase=lowercase,
            pos_label=pos_label
        )

    if args.output:
        save_report(result, args.output)
        print(f"Saved evaluation report to {args.output}")
    else:
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()