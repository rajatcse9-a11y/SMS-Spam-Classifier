import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report)

from src.preprocess import MODEL_DIR, get_split


def main():
    _, X_test, _, y_test = get_split()
    model = joblib.load(MODEL_DIR / "model.pkl")
    tfidf = joblib.load(MODEL_DIR / "vectorizer.pkl")

    pred = model.predict(tfidf.transform(X_test))
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["ham", "spam"], cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved models/confusion_matrix.png")


if __name__ == "__main__":
    main()