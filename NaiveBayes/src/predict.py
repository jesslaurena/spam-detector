# src/predict.py
"""
Predict CLI for the Multinomial Naive Bayes spam detector.

Usage:
  # Single text
  python src/predict.py --text "Buy cheap meds now" --model models/nb_model.pkl --vocab models/vocab.json

  # Batch (one example per line)
  python src/predict.py --input-file data/sample_texts.txt --model models/nb_model.pkl --vocab models/vocab.json --output-file preds.json

Outputs:
  Printed prediction(s) and (optionally) saved JSON with fields:
    text, label, probs (dict class->probability)

Notes:
  - Attempts to import src.preprocess.preprocess and will use it if available.
  - Falls back to a simple tokenizer/preprocessor if not present.
  - Expects vocab saved as JSON (token->index) produced by src.vectorize.save_vocab.
  - Expects model saved by model.save (pickle), loaded via model.MultinomialNB.load.
"""
import argparse
import json
import os
from typing import Dict, Any, Iterable, List, Tuple

# Project imports
from src.vectorize import load_vocab, vectorize
from src.model import MultinomialNB

# Fallback preprocess/tokenize if src.preprocess is not available.
def _fallback_preprocess(text: str, lowercase: bool = True) -> List[str]:
    import re
    if text is None:
        return []
    t = str(text)
    if lowercase:
        t = t.lower()
    return re.findall(r"[a-z0-9]+", t)

def load_artifacts(model_path: str, vocab_path: str) -> Tuple[MultinomialNB, Dict[str, int]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    model = MultinomialNB.load(model_path)
    vocab = load_vocab(vocab_path)
    return model, vocab

def predict_single_text(
    text: str,
    model: MultinomialNB,
    vocab: Dict[str, int],
    preprocess_fn,
    ngram_range=(1,1),
    lowercase: bool = True,
) -> Dict[str, Any]:
    tokens = preprocess_fn(text, lowercase) if preprocess_fn is not None else _fallback_preprocess(text, lowercase)
    x = vectorize(tokens, vocab, ngram_range=tuple(ngram_range), lowercase=lowercase)
    label = model.predict(x)
    probs = model.predict_proba(x)
    return {"text": text, "label": label, "probs": probs}

def predict_multiple_texts(
    texts: Iterable[str],
    model: MultinomialNB,
    vocab: Dict[str, int],
    preprocess_fn,
    ngram_range=(1,1),
    lowercase: bool = True,
) -> List[Dict[str, Any]]:
    results = []
    for t in texts:
        results.append(predict_single_text(t, model, vocab, preprocess_fn, ngram_range=ngram_range, lowercase=lowercase))
    return results

def main():
    parser = argparse.ArgumentParser(description="Predict with trained Naive Bayes spam model")
    parser.add_argument("--model", default="models/nb_model.pkl", help="Path to saved model (pickle)")
    parser.add_argument("--vocab", default="models/vocab.json", help="Path to saved vocab (JSON)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text to classify (wrap in quotes)")
    group.add_argument("--input-file", help="Path to input file (one text per line or JSONL with {\"text\":...})")
    parser.add_argument("--output-file", help="Path to save predictions as JSON or JSONL (if not provided prints to stdout)")
    parser.add_argument("--ngram-range", nargs=2, type=int, default=[1, 1], help="ngram range min max (default 1 1)")
    parser.add_argument("--lowercase", action="store_true", help="lowercase texts before tokenization (default false if not set)")
    args = parser.parse_args()

    # load model and vocab
    try:
        model, vocab = load_artifacts(args.model, args.vocab)
    except Exception as e:
        raise SystemExit(f"Failed to load artifacts: {e}")

    # Try to import team preprocess if available
    try:
        from src.preprocess import preprocess  # type: ignore
        preprocess_fn = preprocess
    except Exception:
        preprocess_fn = None  # we will use fallback via predict_single_text

    ngram_range = (int(args.ngram_range[0]), int(args.ngram_range[1]))
    lowercase = bool(args.lowercase)

    # Prepare input texts
    texts = []
    if args.text:
        texts = [args.text]
    else:
        # read file: accept plain text (one per line) or JSONL where each line is {"text": "..."}
        if not os.path.exists(args.input_file):
            raise SystemExit(f"Input file not found: {args.input_file}")
        with open(args.input_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # try parse JSON line
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "text" in obj:
                            texts.append(str(obj["text"]))
                            continue
                    except Exception:
                        pass
                # fallback: treat entire line as text
                texts.append(line)

    # Run predictions
    results = predict_multiple_texts(texts, model, vocab, preprocess_fn or _fallback_preprocess, ngram_range=ngram_range, lowercase=lowercase)

    # Output
    if args.output_file:
        # If multiple results, save as JSONL (one JSON per line). If single result, save a single JSON object.
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
        # pretty print to stdout
        for r in results:
            print(json.dumps(r, ensure_ascii=False))

if __name__ == "__main__":
    main()