import math
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .result-card {
        padding: 1.2rem 1.5rem; border-radius: 14px; margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .spam-card {background: rgba(239,68,68,0.15); border-color: rgba(239,68,68,0.6);}
    .ham-card  {background: rgba(34,197,94,0.15); border-color: rgba(34,197,94,0.6);}
    .result-title {font-size: 1.6rem; font-weight: 700; margin: 0;}
    .result-sub {opacity: 0.8; margin: 0.2rem 0 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "msg" not in st.session_state:
    st.session_state.msg = ""

EXAMPLES = {
    "🚨 Spam: prize": "Congratulations! You won a free prize. Call now to claim",
    "🚨 Spam: urgent": "URGENT! Your mobile number has won a 2000 pounds bonus. Call 09061701461 now",
    "🚨 Spam: link": "You have been selected for a free cash reward. Click http://claim-now.com",
    "✅ Ham: class": "Can you send me the notes of today's class?",
    "✅ Ham: casual": "Ok I will call you after dinner",
}


def set_example(text: str):
    st.session_state.msg = text


def predict(text: str, url: str) -> dict:
    resp = requests.post(url, json={"text": text}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def confidence(score: float) -> float:
    # decision_function score ko 0-1 range me map karta hai (approx, true probability nahi)
    return 1 / (1 + math.exp(-2 * abs(score)))


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", DEFAULT_API)

    try:
        health_url = api_url.rsplit("/predict", 1)[0] + "/"
        ok = requests.get(health_url, timeout=3).status_code == 200
    except requests.RequestException:
        ok = False
    st.markdown("**API status:** " + ("🟢 Online" if ok else "🔴 Offline"))

    st.divider()
    st.subheader("Example messages")
    for label, text in EXAMPLES.items():
        st.button(label, on_click=set_example, args=(text,), use_container_width=True)

    st.divider()
    st.caption("Model: TF-IDF + LinearSVM\n\nTrained on English SMS only.")

# ---------- Header ----------
st.title("🛡️ SMS Spam Detector")
st.caption("Message paste karo aur check karo ki spam hai ya nahi.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Check", "📁 Batch (CSV)", "🕘 History"])

# ---------- Tab 1: Single ----------
with tab1:
    text = st.text_area("Message", key="msg", height=140, placeholder="Yahan message likho...")
    c1, c2, c3 = st.columns(3)
    c1.caption(f"Characters: {len(text)}")
    c2.caption(f"Words: {len(text.split())}")

    if st.button("Check Message", type="primary"):
        if not text.strip():
            st.warning("Pehle message likho.")
        else:
            try:
                with st.spinner("Checking..."):
                    r = predict(text, api_url)
                label, score = r["label"], r["score"]
                conf = confidence(score)

                css_class = "spam-card" if label == "spam" else "ham-card"
                title = "🚨 SPAM" if label == "spam" else "✅ NOT SPAM"
                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <p class="result-title">{title}</p>
                        <p class="result-sub">Raw score: {score}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(conf, text=f"Confidence (approx): {conf:.0%}")

                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "message": text[:120],
                        "label": label,
                        "score": score,
                    },
                )
            except requests.RequestException as e:
                st.error(f"API se connect nahi ho paya: {e}")

# ---------- Tab 2: Batch ----------
with tab2:
    st.write("CSV upload karo jisme ek text column ho.")
    file = st.file_uploader("CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file, encoding="latin-1")
        st.dataframe(df.head(), use_container_width=True)
        col = st.selectbox("Text column chuno", df.columns)

        if st.button("Run Batch Prediction"):
            labels, scores = [], []
            bar = st.progress(0.0)
            try:
                for i, t in enumerate(df[col].astype(str)):
                    r = predict(t, api_url)
                    labels.append(r["label"])
                    scores.append(r["score"])
                    bar.progress((i + 1) / len(df))

                out = df.copy()
                out["prediction"] = labels
                out["score"] = scores

                m1, m2, m3 = st.columns(3)
                m1.metric("Total", len(out))
                m2.metric("Spam", labels.count("spam"))
                m3.metric("Spam %", f"{labels.count('spam') / len(out):.1%}")

                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "⬇️ Download results",
                    out.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                )
            except requests.RequestException as e:
                st.error(f"API error: {e}")

# ---------- Tab 3: History ----------
with tab3:
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        s = (hist["label"] == "spam").sum()
        h1, h2 = st.columns(2)
        h1.metric("Spam", int(s))
        h2.metric("Not spam", int(len(hist) - s))
        st.dataframe(hist, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Abhi tak koi check nahi kiya.")