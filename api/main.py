import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocess import MODEL_DIR, clean

app = FastAPI(title="Spam Detector")
model = joblib.load(MODEL_DIR / "model.pkl")
tfidf = joblib.load(MODEL_DIR / "vectorizer.pkl")


class Msg(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(msg: Msg):
    X = tfidf.transform([clean(msg.text)])
    score = float(model.decision_function(X)[0])
    return {"label": "spam" if score > 0 else "ham", "score": round(score, 3)}