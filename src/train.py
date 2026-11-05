import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import MODEL_DIR, get_split


def main():
    X_train, X_test, y_train, y_test = get_split()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    models = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LinearSVM": LinearSVC(class_weight="balanced"),
    }

    best_name, best_model, best_f1 = None, None, -1
    for name, m in models.items():
        m.fit(Xtr, y_train)
        f1 = f1_score(y_test, m.predict(Xte))
        print(f"{name}: F1 = {f1:.3f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, m, f1

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    joblib.dump(tfidf, MODEL_DIR / "vectorizer.pkl")
    print(f"\nSaved best model: {best_name} (F1 = {best_f1:.3f})")


if __name__ == "__main__":
    main()