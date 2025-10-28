import gdown
import os

# Create a local data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Map file names to their Google Drive file IDs
files = {
    "bow_vocabulary.json": "1FRqclmrr0wJpHO0UVdfDKLoLw4q7hyr8",
    "spam_assassin_preprocessed.csv": "1-1I8tA7XyFpMDhWtGxa5aoUMg6VGVk5y",
    "spam_assassin.csv": "1XHdG11nTRgNO4c79Ua3DsvunPpJ_FDLf",
    "test_bow_features.csv": "1OrTOMFUUI6-fMxyWtMyEol5Sbv3VezbM",
    "test_tfidf_features.csv": "1lEDDSAbF9BL0caqDD85SW318Codx9V4Y",
    "tfidf_vocabulary.json": "1mMAPlnVUeuHlNddZ4eH5ztiS90QHO6XI",
    "train_bow_features.csv": "1CIrdr2eyTbHMQeoFiR-nlz-IH1MAmlJh",
    "train_tfidf_features.csv": "1CIrdr2eyTbHMQeoFiR-nlz-IH1MAmlJh",
}

for name, file_id in files.items():
    output_path = f"data/{name}"
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {name}...")
    gdown.download(url, output_path, quiet=False)
