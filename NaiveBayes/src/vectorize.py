"""Simple vectorization utilities for text classification."""

from collections import Counter, OrderedDict
import re
import json
from typing import List, Dict, Tuple, Union, Iterable, Optional

TokenList = List[str]
Vocab = Dict[str, int]
SparseVec = Dict[int, int]

# Tokenization / n-grams
_token_pattern = re.compile(r"[a-z0-9]+")


def tokenize(text: str, lowercase: bool = True) -> TokenList:
    if lowercase:
        text = text.lower()
    return _token_pattern.findall(text)


def make_ngrams(tokens: TokenList, n: int) -> List[str]:
    if n <= 1:
        return tokens[:]
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# Vocabulary building
def build_vocab(
    docs: Iterable[Union[str, TokenList]],
    min_freq: int = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: Optional[int] = None,
    lowercase: bool = True
) -> Vocab:
    """
    Build token -> index mapping.
    - docs: iterable of token lists or raw strings
    - min_freq: min corpus frequency to keep a token
    - ngram_range: (min_n, max_n)
    - max_features: keep top-N tokens if set
    """
    counter = Counter()
    min_n, max_n = ngram_range
    for doc in docs:
        if isinstance(doc, str):
            tokens = tokenize(doc, lowercase=lowercase)
        else:
            tokens = [t.lower() if lowercase else t for t in doc]
        for n in range(min_n, max_n + 1):
            counter.update(make_ngrams(tokens, n))

    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_features is not None:
        items = items[:max_features]
    vocab = OrderedDict((tok, i) for i, (tok, _) in enumerate(items))
    return vocab


# Vectorization
def vectorize(
    tokens_or_text: Union[str, TokenList],
    vocab: Vocab,
    ngram_range: Tuple[int, int] = (1, 1),
    lowercase: bool = True
) -> SparseVec:
    """
    Convert one document (tokens or raw string) to sparse counts {index: count}.
    Unknown tokens are ignored.
    """
    if isinstance(tokens_or_text, str):
        tokens = tokenize(tokens_or_text, lowercase=lowercase)
    else:
        tokens = [t.lower() if lowercase else t for t in tokens_or_text]

    counts: SparseVec = {}
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for tok in make_ngrams(tokens, n):
            idx = vocab.get(tok)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
    return counts


def docs_to_matrix(
    docs: Iterable[Union[str, TokenList]],
    vocab: Vocab,
    ngram_range: Tuple[int, int] = (1, 1),
    dense: bool = False,
    dtype=int,
    lowercase: bool = True
) -> Union[List[SparseVec], List[List[int]]]:
    """
    Convert many docs into a list of sparse dicts (default) or dense rows.
    Dense rows have length = len(vocab).
    """
    rows = []
    if dense:
        V = len(vocab)
        for doc in docs:
            sparse = vectorize(doc, vocab, ngram_range=ngram_range, lowercase=lowercase)
            row = [dtype(0)] * V
            for idx, cnt in sparse.items():
                row[idx] = dtype(cnt)
            rows.append(row)
    else:
        for doc in docs:
            rows.append(vectorize(doc, vocab, ngram_range=ngram_range, lowercase=lowercase))
    return rows


# Save / Load vocab
def save_vocab(vocab: Vocab, path: str) -> None:
    with open(path, "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Vocab:
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    items = sorted(data.items(), key=lambda kv: int(kv[1]))
    return OrderedDict((tok, int(idx)) for tok, idx in items)