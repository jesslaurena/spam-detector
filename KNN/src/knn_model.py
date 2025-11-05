"""
K-Nearest Neighbors (from-scratch, NumPy only).
Provides: load_features, build_knn_index, knn_predict_single,
knn_predict_batch, get_k_nearest, predict_label, predict_all.
"""

from typing import List, Tuple, Dict, Any
import csv
import numpy as np

# I/O helpers
def load_features(filename: str, n_features: int = None, has_header: bool = True) -> Tuple[List[List[float]], List[int]]:
    """
    Load a dense CSV with rows: label, feat_0, feat_1, ...
    Returns (features_list, labels_list).
    """
    features: List[List[float]] = []
    labels: List[int] = []

    with open(filename, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)

        for row in reader:
            if not row:
                continue

            # parse label
            try:
                lbl = int(float(row[0]))
            except Exception:
                lbl = 1 if str(row[0]).strip().lower() in ("1", "spam", "true", "t") else 0
            labels.append(lbl)

            feats = row[1:1 + n_features] if n_features is not None else row[1:]
            if n_features is not None and len(feats) < n_features:
                feats += ["0"] * (n_features - len(feats))

            row_feats: List[float] = []
            for v in feats:
                try:
                    row_feats.append(float(v))
                except Exception:
                    row_feats.append(0.0)
            features.append(row_feats)

    return features, labels

# KNN core (NumPy-based)
def build_knn_index(X_train: Any) -> np.ndarray:
    """L2-normalize rows of the training matrix (for cosine similarity)."""
    X = np.asarray(X_train, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    nonzero = norms > 0
    X_norm = np.zeros_like(X, dtype=np.float64)
    if np.any(nonzero):
        X_norm[nonzero] = X[nonzero] / norms[nonzero, None]
    return X_norm

def knn_topk_indices(X_train_norm: np.ndarray, x_test_norm: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k nearest neighbors (by cosine similarity)."""
    sims = X_train_norm.dot(x_test_norm)
    n = sims.size
    if k >= n:
        return np.argsort(-sims)
    part_idx = np.argpartition(-sims, k-1)[:k]
    topk_sorted = part_idx[np.argsort(-sims[part_idx])]
    return topk_sorted

def knn_predict_single(X_train_norm: np.ndarray, y_train: np.ndarray, x_test: List[float], k: int) -> Dict[str, Any]:
    """
    Predict label and confidence for one dense test vector.
    Returns {"label": int, "probs": {"0": p0, "1": p1}}.
    """
    x = np.asarray(x_test, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm > 0:
        x_norm = x / norm
    else:
        labels, counts = np.unique(y_train, return_counts=True)
        maj = int(labels[np.argmax(counts)])
        probs = {"0": 0.0, "1": 0.0}
        probs[str(maj)] = 1.0
        return {"label": maj, "probs": probs}

    idx_topk = knn_topk_indices(X_train_norm, x_norm, k)
    neighbor_labels = y_train[idx_topk]
    ones = int((neighbor_labels == 1).sum())
    zeros = int((neighbor_labels == 0).sum())

    if zeros > ones:
        label = 0
    elif ones > zeros:
        label = 1
    else:
        label = int(neighbor_labels[0])

    conf = max(zeros, ones) / float(k) if k > 0 else 0.0
    probs = {"0": 0.0, "1": 0.0}
    probs[str(label)] = float(conf)
    return {"label": int(label), "probs": probs}

def knn_predict_batch(X_train_norm: np.ndarray, y_train: np.ndarray, X_test: Any, k: int, batch_size: int = 64) -> List[Dict[str, Any]]:
    """
    Predict labels for multiple dense test rows in batches.
    Returns list of dicts consistent with knn_predict_single.
    """
    Xt = np.asarray(X_test, dtype=np.float64)
    n_test = Xt.shape[0]
    results: List[Dict[str, Any]] = []

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        Xb = Xt[start:end]

        norms = np.linalg.norm(Xb, axis=1)
        nonzero = norms > 0
        Xb_norm = np.zeros_like(Xb, dtype=np.float64)
        if np.any(nonzero):
            Xb_norm[nonzero] = Xb[nonzero] / norms[nonzero, None]

        sims = Xb_norm.dot(X_train_norm.T)

        for i in range(sims.shape[0]):
            row = sims[i]
            if k >= row.size:
                idx_topk = np.argsort(-row)
            else:
                part_idx = np.argpartition(-row, k-1)[:k]
                idx_topk = part_idx[np.argsort(-row[part_idx])]

            neighbor_labels = y_train[idx_topk]
            ones = int((neighbor_labels == 1).sum())
            zeros = int((neighbor_labels == 0).sum())

            if zeros > ones:
                label = 0
            elif ones > zeros:
                label = 1
            else:
                label = int(neighbor_labels[0])

            conf = max(zeros, ones) / float(k) if k > 0 else 0.0
            probs = {"0": 0.0, "1": 0.0}
            probs[str(label)] = float(conf)
            results.append({"label": int(label), "probs": probs})

    return results

# Compatibility wrappers (original API)
def get_k_nearest(train_features: List[List[float]], train_labels: List[int], test_email: List[float], k: int) -> List[int]:
    """Return labels of k nearest neighbors (plain Python list)."""
    X = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.int32)
    X_norm = build_knn_index(X)

    x = np.asarray(test_email, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm > 0:
        x_norm = x / norm
    else:
        return list(y[:k].tolist())

    idx_topk = knn_topk_indices(X_norm, x_norm, k)
    return list(y[idx_topk].tolist())

def predict_label(train_features: List[List[float]], train_labels: List[int], test_email: List[float], k: int) -> int:
    """Return predicted label (0 or 1) for a single test email."""
    X = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.int32)
    X_norm = build_knn_index(X)
    res = knn_predict_single(X_norm, y, test_email, k)
    return int(res["label"])

def predict_all(train_features: List[List[float]], train_labels: List[int], test_features: List[List[float]], k: int) -> List[int]:
    """Predict labels for an entire test set and return list of ints."""
    X = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.int32)
    X_norm = build_knn_index(X)
    results = knn_predict_batch(X_norm, y, np.asarray(test_features, dtype=np.float64), k, batch_size=64)
    return [int(r["label"]) for r in results]

# Public API
__all__ = [
    "load_features",
    "build_knn_index",
    "knn_predict_single",
    "knn_predict_batch",
    "get_k_nearest",
    "predict_label",
    "predict_all"
]