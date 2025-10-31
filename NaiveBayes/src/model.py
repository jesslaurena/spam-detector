# src/model.py
"""
Multinomial Naive Bayes classifier that accepts sparse count vectors.
- Input format: a single example is a dict {feature_index: count}.
- Training input: list of sparse dicts and list of labels (strings).
- Model stores class priors and log-probabilities for features (with Laplace smoothing).
- Save/load via pickle for easy reuse by train/predict scripts.
"""

from typing import List, Dict, Any, Iterable, Optional
import numpy as np
import pickle
from math import log
from collections import defaultdict

SparseVec = Dict[int, int]

class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        """
        alpha: Laplace smoothing parameter (>= 0). Typical default is 1.0.
        """
        self.alpha: float = float(alpha)
        self.classes_: List[Any] = []
        self.class_count_: Dict[Any, int] = {}
        self.feature_count_: Optional[np.ndarray] = None  # shape (n_classes, vocab_size)
        self.class_log_prior_: Optional[np.ndarray] = None  # shape (n_classes,)
        self.feature_log_prob_: Optional[np.ndarray] = None  # shape (n_classes, vocab_size)
        self.vocab_size_: int = 0

    def _infer_vocab_size(self, X_sparse: Iterable[SparseVec]) -> int:
        max_idx = -1
        for x in X_sparse:
            if x:
                local_max = max(x.keys())
                if local_max > max_idx:
                    max_idx = local_max
        return max_idx + 1 if max_idx >= 0 else 0

    def fit(self, X_sparse: List[SparseVec], y: List[Any]) -> "MultinomialNB":
        """
        Fit the Multinomial Naive Bayes model.
        X_sparse: list of dicts {feature_index: count}
        y: list of class labels (strings or hashable objects), same length as X_sparse
        """
        if len(X_sparse) != len(y):
            raise ValueError("X_sparse and y must be the same length")

        # determine classes and mapping
        classes = sorted(set(y))
        self.classes_ = classes
        label_to_i = {label: i for i, label in enumerate(classes)}
        n_classes = len(classes)

        # determine vocab size from training data
        self.vocab_size_ = self._infer_vocab_size(X_sparse)

        # initialize counts
        class_doc_counts = defaultdict(int)
        feature_counts = [defaultdict(int) for _ in range(n_classes)]

        # accumulate counts
        for x, label in zip(X_sparse, y):
            idx = label_to_i[label]
            class_doc_counts[label] += 1
            for fidx, cnt in x.items():
                if fidx >= 0:
                    feature_counts[idx][fidx] += cnt

        # convert to numpy arrays
        self.class_count_ = {c: class_doc_counts.get(c, 0) for c in classes}
        total_docs = sum(self.class_count_.values()) or 1

        # feature_count_ as dense array shape (n_classes, vocab_size)
        self.feature_count_ = np.zeros((n_classes, self.vocab_size_), dtype=np.float64)
        for i, _ in enumerate(classes):
            for fidx, cnt in feature_counts[i].items():
                if 0 <= fidx < self.vocab_size_:
                    self.feature_count_[i, fidx] = cnt

        # compute class log priors
        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            self.class_log_prior_[i] = log(self.class_count_.get(c, 0) / total_docs) if self.class_count_.get(c, 0) > 0 else float("-inf")

        # compute feature log probabilities with Laplace smoothing
        self.feature_log_prob_ = np.zeros_like(self.feature_count_)
        for i in range(n_classes):
            counts = self.feature_count_[i]
            denom = counts.sum() + self.alpha * max(1, self.vocab_size_)
            # guard against zero vocab_size
            if self.vocab_size_ == 0:
                self.feature_log_prob_[i, :] = float("-inf")
            else:
                self.feature_log_prob_[i, :] = np.log((counts + self.alpha) / denom)

        return self

    def _log_posterior(self, x_sparse: SparseVec) -> np.ndarray:
        """
        Compute log posterior (log unnormalized) for each class given a sparse input.
        Returns array shape (n_classes,)
        """
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise ValueError("Model not fitted yet")

        logs = self.class_log_prior_.copy()  # shape (n_classes,)
        # sum feature log-prob * count
        for i in range(len(self.classes_)):
            row = self.feature_log_prob_[i]
            s = 0.0
            for fidx, cnt in x_sparse.items():
                if 0 <= fidx < self.vocab_size_:
                    s += row[fidx] * cnt
                # ignore unknown indices
            logs[i] += s
        return logs

    def predict_proba(self, x_sparse: SparseVec) -> Dict[Any, float]:
        """
        Return a dict mapping class -> probability for a single example.
        Uses a stable softmax on log posteriors.
        """
        logs = self._log_posterior(x_sparse)
        # stable softmax
        max_log = np.max(logs)
        exps = np.exp(logs - max_log)
        probs = exps / exps.sum() if exps.sum() > 0 else np.ones_like(exps) / len(exps)
        return {self.classes_[i]: float(probs[i]) for i in range(len(self.classes_))}

    def predict(self, x_sparse: SparseVec) -> Any:
        """
        Predict the class label for a single sparse input.
        """
        logs = self._log_posterior(x_sparse)
        idx = int(np.argmax(logs))
        return self.classes_[idx]

    def predict_batch(self, X_sparse: Iterable[SparseVec]) -> List[Any]:
        """
        Predict labels for an iterable of sparse vectors.
        """
        return [self.predict(x) for x in X_sparse]

    def save(self, path: str) -> None:
        """
        Save the fitted model to disk with pickle.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MultinomialNB":
        """
        Load a saved MultinomialNB model from disk.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, MultinomialNB):
            raise TypeError("Loaded object is not a MultinomialNB instance")
        return obj