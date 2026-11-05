# 🛡️ SpamShield: SMS Spam Detection System

End-to-end SMS spam classifier using **TF-IDF + LinearSVM**, served via a **FastAPI** backend and an interactive **Streamlit** UI, containerized with **Docker** and deployed on **Render**.

🔗 **Live Demo:** <streamlit-app-url>
🔗 **API Docs:** <api-url>/docs

## Features
- Text cleaning pipeline (URL/number normalization, stopword removal)
- Comparison of 3 models: Naive Bayes, Logistic Regression, LinearSVM
- REST API with `/predict` endpoint
- Streamlit UI with single check, batch CSV prediction and history
- Dockerized API, deployed on Render

## Project Structure
```
spam-detector/
├── data/
│   └── spam.csv              # SMS Spam Collection dataset
├── src/
│   ├── preprocess.py         # cleaning, label encoding, train/test split
│   ├── train.py              # TF-IDF + model comparison, saves best model
│   └── evaluate.py           # metrics + confusion matrix
├── models/
│   ├── model.pkl             # trained LinearSVM
│   ├── vectorizer.pkl        # fitted TF-IDF vectorizer
│   └── confusion_matrix.png
├── api/
│   └── main.py               # FastAPI app (/predict)
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset
SMS Spam Collection: ~5,500 messages, ~87% ham and ~13% spam (imbalanced).
Duplicates were removed before splitting to avoid data leakage, and a stratified split was used.

**Insight:** spam messages are ~2x longer on average (~140 vs ~71 characters).

## Model Results
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes | 1.000 | 0.817 | 0.899 |
| Logistic Regression | 0.853 | 0.931 | 0.891 |
| **LinearSVM** | **0.945** | **0.924** | **0.934** |

LinearSVM was selected for the best precision/recall balance. Because the data is imbalanced, F1 on the spam class was used instead of accuracy.

## Setup & Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and evaluate (run from project root)
python -m src.train
python -m src.evaluate

# 3. Start API (terminal 1)
uvicorn api.main:app --reload

# 4. Start UI (terminal 2)
streamlit run app/streamlit_app.py
```

## API Usage
**POST** `/predict`

Request:
```json
{ "text": "Congratulations! You won a free prize. Call now" }
```
Response:
```json
{ "label": "spam", "score": 1.245 }
```

## Docker
```bash
docker build -t spamshield-api .
docker run -p 8000:8000 spamshield-api
```

## Tech Stack
Python, scikit-learn, pandas, FastAPI, Streamlit, Docker, Render

## Limitations
- Trained on English SMS only; Hinglish/Hindi messages may be misclassified.
- Confidence shown in the UI is approximate (LinearSVM does not output true probabilities).

## Future Improvements
- Hinglish/multilingual support
- Transformer model (DistilBERT) comparison
- Probability calibration

## Author
**Rajat**: <GitHub link> | <LinkedIn link>