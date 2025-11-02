# Streamlit Quick Start

## 📦 Installation

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

## 🚢 Deployment to Streamlit Cloud

### Option 1: Streamlit Cloud (Easiest)

1. **Push to GitHub:**
   ```bash
   git add app.py requirements.txt
   git commit -m "Add Streamlit app"
   git push
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Done!** Your app is live and updates automatically on every push.

## 🔧 Customization

### Change Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#your-color"
```

### Add More Features

The app is fully customizable in Python. You can:
- Add more metrics
- Create visualizations
- Add file upload
- Add batch processing
- Export results

## 🐛 Troubleshooting

### "Model file not found"
- Make sure you've trained the Naive Bayes model
- Check that `NaiveBayes/models/nb_model.pkl` exists

### "Import error"
- Install all dependencies: `pip install -r requirements.txt`
- Make sure Python path includes NaiveBayes/src

### "KNN not working"
- This is expected until KNN is implemented
- The app will work fine with just Naive Bayes

## 📝 Adding KNN Support

When KNN is ready:

1. Update `load_knn_model()` in `app.py`
2. Update `predict_knn()` function
3. The UI will automatically show KNN results!

## 🎨 Tips

- Use `st.cache` for loading models (already done)
- Add `st.spinner()` for long operations (already done)
- Use columns for side-by-side layouts (already done)

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)

