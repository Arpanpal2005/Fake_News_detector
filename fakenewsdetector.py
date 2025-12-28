import streamlit as st
import pickle
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ==============================
# GLOBAL CSS
# ==============================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e8f5e9;
    }

    section.main > div {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }

    section[data-testid="stSidebar"] > div {
        background-color: #f1f8e9;
    }

    .stButton>button {
        height: 44px;
        font-size: 16px;
    }

    .animated-word::after {
        content: "Fake";
        color: #c62828;
        font-weight: 700;
        animation: changeWord 5s infinite;
    }

    @keyframes changeWord {
        0%   { content: "Fake"; color: #c62828; }
        50%  { content: "Real"; color: #2e7d32; }
        100% { content: "Fake"; color: #c62828; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# LOAD MODEL & VECTORIZER
# ==============================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_artifacts()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📌 How the System Works")
st.sidebar.markdown(
    """
    1. User enters news text  
    2. Text is vectorized using TF-IDF  
    3. Logistic Regression predicts Fake/Real  
    4. Confidence threshold reduces false positives  
    5. Prediction + confidence shown
    """
)

# ==============================
# HEADER
# ==============================
st.markdown(
    """
    <h1>
        📰 <span class="animated-word"></span> News Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# SESSION STATE
# ==============================
if "news_input" not in st.session_state:
    st.session_state.news_input = ""

def clear_text():
    st.session_state.news_input = ""

# ==============================
# INPUT
# ==============================
news_text = st.text_area(
    "📝 Enter News Text",
    key="news_input",
    height=260,
    placeholder="Paste news content here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_btn = st.button("🔍 Check News")
with col2:
    st.button("🧹 Clear", on_click=clear_text)

# ==============================
# PREDICTION
# ==============================
THRESHOLD = 60

if analyze_btn and news_text.strip():
    vec = vectorizer.transform([news_text])

    prediction = model.predict(vec)[0]  # 0 = Fake, 1 = Real
    probs = model.predict_proba(vec)[0]

    fake_conf = probs[0] * 100
    real_conf = probs[1] * 100
    max_conf = max(fake_conf, real_conf)

    st.markdown("---")

    # ==============================
    # RESULT DISPLAY
    # ==============================
    if max_conf < THRESHOLD:
        st.success("✅ This news is **REAL** (Low Confidence Override)")
    else:
        if prediction == 1:
            st.success("✅ This news is **REAL**")
        else:
            st.error("❌ This news is **FAKE**")

    # ==============================
    # CONFIDENCE DISPLAY (BOTH)
    # ==============================
    c1, c2 = st.columns(2)

    with c1:
        st.metric("🟢 Real Confidence", f"{real_conf:.2f}%")

    with c2:
        st.metric("🔴 Fake Confidence", f"{fake_conf:.2f}%")

elif analyze_btn:
    st.warning("⚠️ Please enter news text.")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<center>Built by <b>Arpan Pal</b> | NLP • Machine Learning • Streamlit</center>",
    unsafe_allow_html=True
)
