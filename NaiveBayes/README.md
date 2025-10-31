# Spam Detector

Simple spam/ham classifier using a Multinomial Naive Bayes implementation and a count-based vectorizer.

Project goal
- Quickly classify incoming messages as spam or ham and provide a confidence score. The project uses a text dataset with columns emailId (int), label (spam/ham), and text (string) and aims for >90% accuracy on the test set [1].

Status
- Core components included: vectorize.py, model.py, train.py, predict.py, evaluate.py, tests/test.py, config.yaml, requirements.txt, .gitignore.
- These files implement a self-contained Naive Bayes pipeline that accepts sparse count-vectors and saves model/vocab artifacts.

Quick start (local)
1. Create a virtual environment and install dependencies:
   - python -m venv .venv
   - source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   - pip install -r requirements.txt

2. Prepare data
   - Put your labeled CSV in `data/` (e.g., `data/spam.csv`).
   - The CSV should contain at least columns `text` and `label`. If your file uses different column names, either rename them or update `src/train.py` to map them.

3. Train
   - Example:
     python src/train.py --data data/spam.csv --config config.yaml --save_dir models
   - Artifacts produced:
     - models/nb_model.pkl (pickled model)
     - models/vocab.json (vocabulary)
     - models/metrics.json (evaluation on the held-out test split)
     - models/train_config.json (config used)

4. Predict single message
   - Example:
     python src/predict.py --text "Buy cheap meds now" --model models/nb_model.pkl --vocab models/vocab.json

5. Evaluate on a labeled CSV
   - Example:
     python src/evaluate.py --data data/test.csv --model models/nb_model.pkl --vocab models/vocab.json --output models/eval_metrics.json

Testing
- Run unit tests:
  - pytest -q

Files and responsibilities (team mapping)
- Jess: dataset loader / cleaning / preprocessing and frontend integration (data loader should produce DataFrame with columns emailId, label, text).  
- Isabella: k-NN implementation and integration (separate module).  
- Suchir: Naive Bayes (vectorize.py, model.py, train/predict/evaluate, tests).  
This distribution of responsibilities was agreed in the project plan [1].

Config
- See config.yaml for default hyperparameters (alpha, min_freq, ngram_range, test_size, random_seed). Modify to run experiments.

Notes & integration points
- The code includes a fallback simple tokenizer when `src.preprocess.preprocess` (Jess’s preprocess) is not available. For best results, agree as a team on a shared `preprocess(text) -> List[str]` API and confirm `ngram_range` / `lowercase` settings.
- The vectorizer saves vocab as JSON (token -> index). Model expects sparse dict inputs {index: count}.
- Aim to tune alpha, min_freq, and n-gram settings to improve performance. The project target is >90% test accuracy when using a suitable public spam dataset [1].

Troubleshooting
- If training errors about missing columns, confirm your CSV column names match `text` and `label`.
- If precision/recall are NaN or zero, check label values (e.g., `spam` vs `spam\r`) and class balance.

License & acknowledgements
- (Add your chosen license and acknowledgements here.)