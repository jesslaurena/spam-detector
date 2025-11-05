"""Train pipeline for the Multinomial Naive Bayes spam detector."""

import os
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

# ensure project package imports work when running this script directly
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.vectorize import build_vocab, docs_to_matrix, save_vocab
from src.model import MultinomialNB

# default config
DEFAULT_CONFIG = {
    "alpha": 1.0,
    "min_freq": 2,
    "ngram_range": [1, 1],
    "max_features": None,
    "test_size": 0.2,
    "random_seed": 42,
    "lowercase": True,
    "pos_label": "spam"
}


def _fallback_preprocess(text: str, lowercase: bool = True) -> List[str]:
    """Minimal tokenizer used if src.preprocess.preprocess is missing."""
    import re
    if text is None:
        return []
    t = str(text)
    if lowercase:
        t = t.lower()
    return re.findall(r"[a-z0-9]+", t)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config if available, otherwise return defaults."""
    if not path:
        return DEFAULT_CONFIG.copy()
    try:
        import yaml
        with open(path, "r", encoding="utf8") as f:
            cfg = yaml.safe_load(f) or {}
        merged = DEFAULT_CONFIG.copy()
        merged.update(cfg)
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV and normalize column names.
    Expect a CSV with 'text' and 'label' (common alternatives are handled).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)

    if 'text' not in df.columns:
        for cand in ['message', 'body', 'Text', 'content', 'email_content', 'emailContent']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'text'})
                break
    if 'label' not in df.columns:
        for cand in ['spam', 'labelled', 'Label']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'label'})
                break

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'text' and 'label' columns")

    df['text'] = df['text'].fillna('').astype(str)
    df['label'] = df['label'].astype(str)
    return df[['text', 'label']]


def train_pipeline(
    df: pd.DataFrame,
    preprocess_fn,
    config: Dict[str, Any],
    save_dir: str = "models"
) -> Dict[str, Any]:
    """Train NB on df, save model/vocab/metrics and return paths + report."""
    os.makedirs(save_dir, exist_ok=True)

    # tokenize
    docs_tokens = [preprocess_fn(t, config.get("lowercase", True)) for t in df['text'].tolist()]

    # build vocab
    ngram_range = tuple(config.get("ngram_range", [1, 1]))
    vocab = build_vocab(
        docs_tokens,
        min_freq=int(config.get("min_freq", 1)),
        ngram_range=ngram_range,
        max_features=config.get("max_features", None),
        lowercase=config.get("lowercase", True)
    )

    # vectorize to sparse dicts
    X_sparse = docs_to_matrix(docs_tokens, vocab, ngram_range=ngram_range, dense=False, lowercase=config.get("lowercase", True))
    y = df['label'].tolist()

    # split using indices so sparse structure is preserved
    indices = list(range(len(X_sparse)))
    stratify = y if len(set(y)) > 1 else None
    test_size = float(config.get("test_size", 0.2))

    if stratify is not None:
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y, test_size=test_size,
            random_state=int(config.get("random_seed", 42)),
            stratify=y
        )
    else:
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y, test_size=test_size,
            random_state=int(config.get("random_seed", 42))
        )

    X_train = [X_sparse[i] for i in train_idx]
    X_test = [X_sparse[i] for i in test_idx]

    # fit model
    model = MultinomialNB(alpha=float(config.get("alpha", 1.0)))
    model.fit(X_train, y_train)

    # evaluate
    y_pred = [model.predict(x) for x in X_test]
    pos_label = config.get("pos_label", DEFAULT_CONFIG["pos_label"])
    try:
        precision = metrics.precision_score(y_test, y_pred, pos_label=pos_label)
        recall = metrics.recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = metrics.f1_score(y_test, y_pred, pos_label=pos_label)
    except Exception:
        precision = metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = metrics.recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)

    report = {
        "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred).tolist(),
        "n_classes": len(set(y)),
        "vocab_size": len(vocab)
    }

    # save artifacts
    model_path = os.path.join(save_dir, "nb_model.pkl")
    vocab_path = os.path.join(save_dir, "vocab.json")
    metrics_path = os.path.join(save_dir, "metrics.json")

    model.save(model_path)
    save_vocab(vocab, vocab_path)
    with open(metrics_path, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)

    cfg_path = os.path.join(save_dir, "train_config.json")
    with open(cfg_path, "w", encoding="utf8") as f:
        json.dump(config, f, indent=2)

    return {
        "model_path": model_path,
        "vocab_path": vocab_path,
        "metrics_path": metrics_path,
        "report": report
    }


def main():
    parser = argparse.ArgumentParser(description="Train Multinomial Naive Bayes spam detector")
    parser.add_argument("--data", required=True, help="Path to CSV file with columns 'text' and 'label'")
    parser.add_argument("--config", default=None, help="Path to YAML config (optional)")
    parser.add_argument("--save_dir", default="models", help="Directory to save model/vocab/metrics")
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        df = load_data(args.data)
    except Exception as e:
        raise SystemExit(f"Failed to load data: {e}")

    try:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from src.preprocess import preprocess  # type: ignore
        preprocess_fn = preprocess
    except Exception:
        preprocess_fn = _fallback_preprocess

    result = train_pipeline(df, preprocess_fn, config, save_dir=args.save_dir)
    print("Training complete. Artifacts saved to:", args.save_dir)
    print("Metrics:", json.dumps(result["report"], indent=2))


if __name__ == "__main__":
    main()