# 📡 BBC News Category Classifier

> Naive Bayes · TF-IDF · Live BBC RSS Feeds · Streamlit · Pickle

A machine learning web app that classifies any news headline or paragraph into one of 6 BBC categories — **Politics, Technology, Business, Sport, Science, Entertainment** — in real time.

---

## Screenshots
<img width="481" height="762" alt="image" src="https://github.com/user-attachments/assets/00427e06-7ad6-4395-9f1e-4d4fcb56132f" />

## Deployment link
https://newsclassifierwebapp-kdzgqkmsbjovkeeqgnhccb.streamlit.app/

## 1. Project Overview

This project demonstrates a complete end-to-end NLP pipeline:

- Fetches live article data from BBC RSS feeds at runtime
- Preprocesses text (lowercasing, punctuation removal, stopword filtering, stemming)
- Vectorises text using TF-IDF with bigrams
- Trains a Multinomial Naive Bayes classifier
- Pickles (saves) the trained model and vectorizer to disk
- Loads the pickled model on subsequent runs — no retraining needed
- Serves a polished dark-themed Streamlit web interface

**Categories covered:**

| 🏛️ Politics | 💻 Technology | 💼 Business | ⚽ Sport | 🔬 Science | 🎬 Entertainment |
|---|---|---|---|---|---|

---

## 2. Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `scikit-learn` | TF-IDF vectorizer + Naive Bayes model |
| `nltk` | Stopwords + PorterStemmer |
| `feedparser` | BBC RSS feed ingestion |
| `pandas` | DataFrame handling |
| `pickle` | Model serialisation (built-in) |

---

## 3. How It Works

### 3.1 NLP Pipeline

Every message goes through the same cleaning function before training or prediction:

```python
def clean(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation
    words = [w for w in text.split()             # remove stopwords
             if w not in stopwords and len(w) > 2]
    words = [PorterStemmer().stem(w)             # stem words
             for w in words]
    return " ".join(words)
```

### 3.2 Model Training

- RSS feeds are parsed using feedparser — title + summary text per article
- Text is cleaned with the pipeline above
- 80/20 train-test split with `random_state=42`
- `TfidfVectorizer` with `max_features=5000` and bigrams `ngram_range=(1,2)`
- `MultinomialNB` with `alpha=0.5` (Laplace smoothing)
- Accuracy is computed on the test split and stored alongside the model

### 3.3 Prediction Flow

```
User types text
     ↓
clean(text)  →  stemmed tokens
     ↓
tfidf.transform([cleaned])  →  feature vector
     ↓
mnb.predict()       →  category label
mnb.predict_proba() →  confidence scores
     ↓
Rendered in UI with coloured confidence bars
```

---

## 4. How Pickling Works

Python's `pickle` module serialises trained objects (model + vectorizer) into binary `.pkl` files on disk so they can be restored instantly without retraining.

### 4.1 Saving the model

```python
import pickle

with open("bbc_model.pkl", "wb") as f: pickle.dump(model, f)
with open("bbc_tfidf.pkl", "wb") as f: pickle.dump(tfidf, f)
with open("bbc_acc.pkl",   "wb") as f: pickle.dump(acc,   f)
with open("bbc_meta.pkl",  "wb") as f: pickle.dump({"n_samples": n}, f)
```

### 4.2 Loading the model

```python
with open("bbc_model.pkl", "rb") as f: model = pickle.load(f)
with open("bbc_tfidf.pkl", "rb") as f: tfidf = pickle.load(f)
with open("bbc_acc.pkl",   "rb") as f: acc   = pickle.load(f)
```

### 4.3 Smart load logic

The app checks for `.pkl` files on startup:

```
if all pickle files exist:
    load from disk  ✅  (instant)
else:
    fetch BBC RSS → train → save pickles  🔄  (~5 seconds)
```

The **⚙️ Model Controls** expander in the app includes a **Retrain & Re-pickle** button that deletes old pickles and forces a fresh training cycle from the latest BBC feeds.

---

## 5. Run Locally

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bbc-classifier.git
cd bbc-classifier
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run it fetches BBC feeds, trains the model, and saves `.pkl` files. Every subsequent run loads from those files instantly.

### Folder structure

```
bbc-classifier/
├── app.py                ← Streamlit app
├── requirements.txt      ← Python dependencies
├── .gitignore            ← excludes .pkl files
├── README.md             ← this file
└── (auto-generated on first run)
    ├── bbc_model.pkl
    ├── bbc_tfidf.pkl
    ├── bbc_acc.pkl
    └── bbc_meta.pkl
```

---

## 6. GitHub — Push Your Code

### Step 1 — Initialise Git

```bash
git init
```

### Step 2 — Create .gitignore

Create a file named `.gitignore` in your project folder:

```
# Pickle files — excluded, app retrains on deploy
bbc_model.pkl
bbc_tfidf.pkl
bbc_acc.pkl
bbc_meta.pkl

# Python
__pycache__/
*.pyc
.env
venv/
```

### Step 3 — Commit and push

Go to `github.com → New Repository`, name it `bbc-classifier`, then run:

```bash
git add .
git commit -m "Initial commit - BBC News Classifier"
git remote add origin https://github.com/YOUR_USERNAME/bbc-classifier.git
git branch -M main
git push -u origin main
```

### Updating later

```bash
git add .
git commit -m "describe your change"
git push
```

Streamlit Cloud auto-detects every push and redeploys within a minute.

---

## 7. Deployment — Streamlit Cloud (Free)

### Step 1 — Sign in

Go to **share.streamlit.io** and sign in with your GitHub account.

### Step 2 — New App

- Click **New app**
- Repository: `YOUR_USERNAME/bbc-classifier`
- Branch: `main`
- Main file path: `app.py`
- Click **Deploy!**

### Step 3 — Your live URL

```
https://YOUR_USERNAME-bbc-classifier-app-xxxx.streamlit.app
```

### What happens on deploy

| Step | What happens |
|---|---|
| 1 | Streamlit reads `requirements.txt` and installs packages |
| 2 | `app.py` runs — no `.pkl` files found |
| 3 | BBC RSS feeds are fetched and parsed |
| 4 | Model trains in ~5 seconds |
| 5 | Pickles saved to session filesystem |
| 6 | App is live and accepting predictions ✅ |

> **Note:** Streamlit Cloud's filesystem resets on each cold start, so the app retrains automatically on every new deployment. This takes only a few seconds and is completely expected.

---

## 8. Possible Improvements

- Add more BBC categories (Health, Education, Travel)
- Replace Naive Bayes with a transformer model (e.g. DistilBERT)
- Add a confidence threshold — show "Uncertain" below 40%
- Store user predictions to a CSV for feedback collection
- Add language detection for non-English inputs
- Schedule automatic retraining with GitHub Actions

---

## Author

**Shreya**  
Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer

*Built with ❤️ using Python · scikit-learn · NLTK · Streamlit*
