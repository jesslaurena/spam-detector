# src/evaluate.py
"""
Evaluation utilities for the spam detector.

Provides:
- compute_metrics(y_true, y_pred, pos_label='spam'): returns accuracy, precision, recall, f1, confusion matrix, classification report.
- evaluate_model_on_df: run model predictions on a DataFrame with 'text' and 'label' columns and return metrics + preds.
- CLI to evaluate a saved model + vocab against a CSV dataset and save a metrics JSON.

Usage (example):
  python src/evaluate.py --data data/test.csv --model models/nb_model.pkl --vocab models/vocab.json --output models/eval_metrics.json
"""
import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Iterable, Optional

import pandas as pd
from sklearn import metrics

# Add parent directory to path so we can import src modules
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.model import MultinomialNB
from src.vectorize import vectorize, load_vocab

# Fallback preprocess if src.preprocess is not available
def _fallback_preprocess(text: str, lowercase: bool = True) -> List[str]:
    import re
    if text is None:
        return []
    t = str(text)
    if lowercase:
        t = t.lower()
    return re.findall(r"[a-z0-9]+", t)

def compute_metrics(y_true: Iterable[str], y_pred: Iterable[str], pos_label: Optional[str] = "spam") -> Dict[str, Any]:
    """
    Compute common classification metrics.
    - If binary and pos_label provided and present in labels, compute precision/recall/f1 for that pos_label.
    - Otherwise compute macro-averaged precision/recall/f1.
    Returns a dictionary suitable for JSON serialization.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))

    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    # Try binary metrics for pos_label if applicable
    precision = recall = f1 = None
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
        # fallback safe values
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
    preprocess_fn = None,
    ngram_range: Tuple[int,int] = (1,1),
    lowercase: bool = True,
    pos_label: Optional[str] = "spam"
) -> Dict[str, Any]:
    """
    Evaluate a saved model on a DataFrame with columns 'text' and 'label'.
    Returns a dict with metrics and predictions.
    """
    if 'text' not in df.columns or 'label' not in df.columns:
        # attempt common alternative column names
        if 'message' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'message': 'text'})
        else:
            raise ValueError("DataFrame must contain 'text' and 'label' columns")

    preprocess = preprocess_fn if preprocess_fn is not None else _fallback_preprocess

    texts = df['text'].fillna('').astype(str).tolist()
    true_labels = df['label'].astype(str).tolist()

    preds = []
    probs = []
    for t in texts:
        tokens = preprocess(t, lowercase) if preprocess is not None else _fallback_preprocess(t, lowercase)
        x = vectorize(tokens, vocab, ngram_range=ngram_range, lowercase=lowercase)
        pred = model.predict(x)
        pred_probs = model.predict_proba(x)
        preds.append(pred)
        probs.append(pred_probs)

    metrics_report = compute_metrics(true_labels, preds, pos_label=pos_label)
    result = {
        "metrics": metrics_report,
        "predictions": [
            {"text": txt, "true_label": true, "pred_label": pred, "probs": prob}
            for txt, true, pred, prob in zip(texts, true_labels, preds, probs)
        ]
    }
    return result

def save_report(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    # normalize column names if possible
    if 'text' not in df.columns:
        for cand in ['message', 'body', 'Text', 'content']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'text'})
                break
    if 'label' not in df.columns:
        for cand in ['spam', 'labelled', 'Label']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'label'})
                break
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")
    return df[['text', 'label']]

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved Naive Bayes model on a labeled CSV")
    parser.add_argument("--data", required=True, help="Path to CSV with columns 'text' and 'label'")
    parser.add_argument("--model", required=True, help="Path to saved model pickle (nb_model.pkl)")
    parser.add_argument("--vocab", required=True, help="Path to vocab JSON (vocab.json)")
    parser.add_argument("--output", default=None, help="Path to save evaluation report JSON")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=[1,1], help="ngram range min max (default 1 1)")
    parser.add_argument("--lowercase", action="store_true", help="lowercase texts before tokenization")
    parser.add_argument("--pos-label", default="spam", help="positive label for binary metrics (default 'spam')")
    args = parser.parse_args()

    # Load artifacts
    try:
        model = MultinomialNB.load(args.model)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")
    try:
        vocab = load_vocab(args.vocab)
    except Exception as e:
        raise SystemExit(f"Failed to load vocab: {e}")

    # Load data
    try:
        df = load_dataset(args.data)
    except Exception as e:
        raise SystemExit(f"Failed to load data: {e}")

    # Try to import team preprocess if available
    try:
        from src.preprocess import preprocess  # type: ignore
        preprocess_fn = preprocess
    except Exception:
        preprocess_fn = None

    ngram_range = tuple(args.ngram_range)
    lowercase = bool(args.lowercase)

    result = evaluate_model_on_df(
        df,
        model,
        vocab,
        preprocess_fn=preprocess_fn,
        ngram_range=ngram_range,
        lowercase=lowercase,
        pos_label=args.pos_label
    )

    if args.output:
        save_report(result, args.output)
        print(f"Saved evaluation report to {args.output}")
    else:
        # print a concise summary
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()