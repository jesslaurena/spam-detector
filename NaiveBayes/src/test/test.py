# test/test.py
# Run with: pytest -q
import json
from pathlib import Path

from src.vectorize import build_vocab, vectorize, docs_to_matrix, save_vocab, load_vocab
from src.model import MultinomialNB

def test_vectorize_basic():
    docs = [
        "Buy cheap meds now",
        "Hello friend how are you",
        "Cheap meds available buy today"
    ]
    vocab = build_vocab(docs, min_freq=1, ngram_range=(1,1))
    # Expect some known tokens
    assert "buy" in vocab
    assert "cheap" in vocab
    # Vectorize first document and check counts
    vec0 = vectorize(docs[0], vocab, ngram_range=(1,1))
    buy_idx = vocab["buy"]
    cheap_idx = vocab["cheap"]
    assert isinstance(vec0, dict)
    assert vec0.get(buy_idx, 0) >= 1
    assert vec0.get(cheap_idx, 0) >= 1

def test_vectorize_ngrams_and_min_freq():
    docs = [
        "cheap meds available",
        "cheap meds today",
        "cheap offer"
    ]
    # bigrams should include "cheap meds"
    vocab_bi = build_vocab(docs, min_freq=1, ngram_range=(1,2))
    assert "cheap meds" in vocab_bi
    # min_freq filtering: token 'offer' appears once, if min_freq=2 it should be removed
    vocab_filtered = build_vocab(docs, min_freq=2, ngram_range=(1,1))
    assert "offer" not in vocab_filtered

def test_docs_to_matrix_dense_and_sparse():
    docs = ["buy cheap", "hello world"]
    vocab = build_vocab(docs, min_freq=1, ngram_range=(1,1))
    sparse_rows = docs_to_matrix(docs, vocab, ngram_range=(1,1), dense=False)
    assert isinstance(sparse_rows, list) and all(isinstance(r, dict) for r in sparse_rows)
    dense_rows = docs_to_matrix(docs, vocab, ngram_range=(1,1), dense=True)
    assert isinstance(dense_rows, list) and all(isinstance(r, list) for r in dense_rows)
    assert len(dense_rows[0]) == len(vocab)
    # indices in sparse rows should be valid
    for r in sparse_rows:
        for idx in r.keys():
            assert 0 <= idx < len(vocab)

def test_save_load_vocab(tmp_path):
    docs = ["a b c", "b c d"]
    vocab = build_vocab(docs, min_freq=1, ngram_range=(1,1))
    p = tmp_path / "vocab_test.json"
    save_vocab(vocab, str(p))
    loaded = load_vocab(str(p))
    # JSON saved indices are ints; ordering should be preserved by load_vocab
    assert dict(loaded) == dict(vocab)

def test_model_train_predict_and_proba():
    # Tiny dataset
    docs = [
        "buy cheap meds",     # spam
        "cheap cheap buy",    # spam
        "hello how are you",  # ham
        "good morning friend" # ham
    ]
    labels = ["spam", "spam", "ham", "ham"]
    vocab = build_vocab(docs, min_freq=1, ngram_range=(1,1))
    X = [vectorize(d, vocab, ngram_range=(1,1)) for d in docs]

    model = MultinomialNB(alpha=1.0)
    model.fit(X, labels)

    # Predict on a spammy and hammy example
    spam_test = vectorize("buy cheap", vocab, ngram_range=(1,1))
    ham_test = vectorize("good morning", vocab, ngram_range=(1,1))
    p_spam = model.predict(spam_test)
    p_ham = model.predict(ham_test)
    assert isinstance(p_spam, str) and isinstance(p_ham, str)
    # Predict_proba returns normalized probabilities summing to ~1
    probs = model.predict_proba(spam_test)
    assert set(probs.keys()) == set(model.classes_)
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 1e-6

    # Unknown token (not in vocab) should not crash the model
    unk = vectorize("this_token_is_not_in_vocab", vocab, ngram_range=(1,1))
    probs_unk = model.predict_proba(unk)
    assert set(probs_unk.keys()) == set(model.classes_)
    # empty sparse vector (no known tokens) should return a probability distribution
    empty = {}
    probs_empty = model.predict_proba(empty)
    assert abs(sum(probs_empty.values()) - 1.0) < 1e-6