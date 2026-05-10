import streamlit as st
import feedparser, re, nltk, pandas as pd, pickle, os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords", quiet=True)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Theme ── */
:root {
    --bbc-red:    #BB1919;
    --bbc-dark:   #0D0D0D;
    --bbc-card:   #161616;
    --bbc-border: #2a2a2a;
    --bbc-muted:  #888888;
    --bbc-white:  #F5F0EB;
    --accent:     #E8C547;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bbc-dark) !important;
    color: var(--bbc-white) !important;
}

.stApp {
    background: var(--bbc-dark) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 760px !important;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #BB1919 0%, #7a0f0f 100%);
    border-radius: 20px;
    padding: 44px 40px 36px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "BBC";
    position: absolute;
    top: -20px;
    right: -10px;
    font-family: 'Playfair Display', serif;
    font-size: 130px;
    font-weight: 900;
    color: rgba(255,255,255,0.06);
    letter-spacing: -4px;
    line-height: 1;
    pointer-events: none;
}
.hero-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 14px;
    color: rgba(255,255,255,0.65);
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 30px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #fff;
    margin-top: 16px;
    backdrop-filter: blur(4px);
}

/* ── Cards ── */
.stat-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    margin-bottom: 28px;
}
.stat-card {
    background: var(--bbc-card);
    border: 1px solid var(--bbc-border);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--bbc-red); }
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 11px;
    color: var(--bbc-muted);
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section label ── */
.section-label {
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--bbc-muted);
    margin-bottom: 10px;
    font-weight: 500;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: var(--bbc-card) !important;
    border: 1px solid var(--bbc-border) !important;
    border-radius: 14px !important;
    color: var(--bbc-white) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 18px !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: var(--bbc-red) !important;
    box-shadow: 0 0 0 3px rgba(187,25,25,0.15) !important;
}
.stTextArea textarea::placeholder { color: #555 !important; }
.stTextArea label { display: none !important; }

/* ── Primary Button ── */
.stButton > button {
    width: 100%;
    background: var(--bbc-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
    padding: 14px 0 !important;
    margin-top: 10px;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #9a1414 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(187,25,25,0.4) !important;
}
.stButton > button:active { transform: translateY(0); }

/* ── Result card ── */
.result-card {
    background: var(--bbc-card);
    border: 1px solid var(--bbc-border);
    border-left: 4px solid var(--bbc-red);
    border-radius: 14px;
    padding: 28px 28px 24px;
    margin: 24px 0 20px;
    animation: slideUp 0.35s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-icon { font-size: 48px; margin-bottom: 10px; }
.result-category {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}
.result-tag {
    display: inline-block;
    background: rgba(187,25,25,0.2);
    border: 1px solid rgba(187,25,25,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #ff8888;
    font-weight: 500;
}

/* ── Confidence section ── */
.conf-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--bbc-muted);
    margin: 20px 0 14px;
    font-weight: 500;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.conf-cat {
    font-size: 13px;
    color: var(--bbc-white);
    width: 110px;
    flex-shrink: 0;
}
.conf-bar-wrap {
    flex: 1;
    background: #222;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}
.conf-pct {
    font-size: 12px;
    color: var(--bbc-muted);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Warning ── */
.stAlert {
    background: #1c1300 !important;
    border: 1px solid #3d2b00 !important;
    border-radius: 12px !important;
    color: var(--bbc-white) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bbc-card) !important;
    border: 1px solid var(--bbc-border) !important;
    border-radius: 12px !important;
    font-size: 13px !important;
    color: var(--bbc-muted) !important;
}
.streamlit-expanderContent {
    background: var(--bbc-card) !important;
    border: 1px solid var(--bbc-border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── Divider ── */
hr { border-color: var(--bbc-border) !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 12px;
    color: #444;
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid var(--bbc-border);
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
BBC_FEEDS = {
    "Politics":      "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "Technology":    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business":      "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Sport":         "https://feeds.bbci.co.uk/sport/rss.xml",
    "Science":       "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "Entertainment": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
}
ICONS = {
    "Politics": "🏛️", "Technology": "💻", "Business": "💼",
    "Sport": "⚽",    "Science": "🔬",    "Entertainment": "🎬"
}
BAR_COLORS = {
    "Politics": "#e05252", "Technology": "#52a8e0", "Business": "#52c987",
    "Sport":    "#e0a852", "Science":    "#a852e0", "Entertainment": "#e05298"
}

MODEL_PATH = "bbc_model.pkl"
TFIDF_PATH = "bbc_tfidf.pkl"
ACC_PATH   = "bbc_acc.pkl"
META_PATH  = "bbc_meta.pkl"

sw = set(stopwords.words("english"))
ps = PorterStemmer()

def clean(t):
    t = re.sub(r"[^a-z\s]", " ", t.lower())
    return " ".join(ps.stem(w) for w in t.split() if w not in sw and len(w) > 2)

def save_model(model, tfidf, acc, n_samples):
    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(TFIDF_PATH, "wb") as f: pickle.dump(tfidf, f)
    with open(ACC_PATH,   "wb") as f: pickle.dump(acc,   f)
    with open(META_PATH,  "wb") as f: pickle.dump({"n_samples": n_samples}, f)

def load_from_pickle():
    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f: tfidf = pickle.load(f)
    with open(ACC_PATH,   "rb") as f: acc   = pickle.load(f)
    with open(META_PATH,  "rb") as f: meta  = pickle.load(f)
    return model, tfidf, acc, meta["n_samples"]

@st.cache_resource(show_spinner="📡 Fetching live BBC feeds and training model...")
def load_model():
    pickles = [MODEL_PATH, TFIDF_PATH, ACC_PATH, META_PATH]
    if all(os.path.exists(p) for p in pickles):
        model, tfidf, acc, n_samples = load_from_pickle()
        return model, tfidf, acc, n_samples, "pickle"

    rows = []
    for cat, url in BBC_FEEDS.items():
        for e in feedparser.parse(url).entries:
            text = e.get("title", "") + " " + e.get("summary", "")
            if text.strip():
                rows.append({"text": text, "label": cat})

    df = pd.DataFrame(rows)
    df["clean"] = df["text"].apply(clean)
    n_samples = len(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    model = MultinomialNB(alpha=0.5)
    model.fit(tfidf.fit_transform(X_tr), y_tr)
    acc = accuracy_score(y_te, model.predict(tfidf.transform(X_te)))

    save_model(model, tfidf, acc, n_samples)
    return model, tfidf, acc, n_samples, "trained"

model, tfidf, acc, n_samples, source = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
source_text = "Loaded from saved model" if source == "pickle" else "Trained on live BBC RSS"
st.markdown(f"""
<div class="hero">
    <div class="hero-label">AI · Natural Language Processing</div>
    <div class="hero-title">BBC News<br>Classifier</div>
    <div class="hero-sub">Paste any headline or paragraph — get the category instantly</div>
    <div class="hero-badge">📡 {source_text}</div>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-number">{acc*100:.1f}%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{n_samples}</div>
        <div class="stat-label">Articles</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">6</div>
        <div class="stat-label">Categories</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Enter News Text</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="news_input",
    height=140,
    placeholder="e.g.  Scientists discover a new treatment for Alzheimer's disease using AI-driven drug analysis...",
    label_visibility="collapsed"
)

classify_btn = st.button("🔍  Classify News", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if classify_btn:
    if not user_input.strip():
        st.warning("⚠️  Please enter some text before classifying.")
    else:
        cleaned     = clean(user_input)
        transformed = tfidf.transform([cleaned])
        pred        = model.predict(transformed)[0]
        proba       = model.predict_proba(transformed)[0]
        scores      = dict(zip(model.classes_, (proba * 100)))
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_conf    = scores[pred]

        # Build confidence bars HTML separately (avoids nested f-string escaping bug)
        bars_html = ""
        for cat, score in sorted_scores:
            color = BAR_COLORS.get(cat, "#888")
            bars_html += (
                f'<div class="conf-row">'
                f'  <div class="conf-cat">{ICONS[cat]} {cat}</div>'
                f'  <div class="conf-bar-wrap">'
                f'    <div class="conf-bar" style="width:{score:.1f}%;background:{color};"></div>'
                f'  </div>'
                f'  <div class="conf-pct">{score:.1f}%</div>'
                f'</div>'
            )

        # Result card
        result_html = (
            f'<div class="result-card">'
            f'  <div class="result-icon">{ICONS[pred]}</div>'
            f'  <div class="result-category">{pred}</div>'
            f'  <span class="result-tag">✓ {top_conf:.1f}% confidence</span>'
            f'  <div class="conf-label" style="margin-top:20px;">Confidence breakdown</div>'
            f'  {bars_html}'
            f'</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️  Model Controls"):
    st.markdown(
        '<span style="color:#888;font-size:13px;">Delete saved model files and retrain from the latest BBC RSS feeds.</span>',
        unsafe_allow_html=True
    )
    if st.button("🔄  Retrain & Re-pickle Model"):
        for path in [MODEL_PATH, TFIDF_PATH, ACC_PATH, META_PATH]:
            if os.path.exists(path): os.remove(path)
        st.cache_resource.clear()
        st.success("Model files deleted — reloading fresh...")
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Naive Bayes · TF-IDF · BBC RSS Feeds · Built with Streamlit
</div>
""", unsafe_allow_html=True)