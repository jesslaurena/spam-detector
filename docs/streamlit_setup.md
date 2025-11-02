# Streamlit Quick Start

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install Streamlit separately:
```bash
pip install streamlit
```

### 2. Train Your Models

Make sure your models are trained first:

```bash
# Train Naive Bayes
cd NaiveBayes
python src/train.py --data ../data/spam_assassin_preprocessed.csv --config config.yaml --save_dir models
cd ..
```

TODO: Add KNN

### 3. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### "Import error"
- Install all dependencies: `pip install -r requirements.txt`
- Make sure Python path includes NaiveBayes/src

## Adding KNN Support

When KNN is ready:

1. Update `load_knn_model()` in `app.py`
2. Update `predict_knn()` function
3. The UI will automatically show KNN results

## Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)

