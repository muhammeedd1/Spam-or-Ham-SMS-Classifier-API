from pathlib import Path

import joblib
import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.preprocessing import preprocess


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "spam.csv"
MODELS_DIR = BASE_DIR / "models"

TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "Best_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "Label_encoder.pkl"


def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)


def read_data(data_path):
    data = pd.read_csv(data_path, encoding="ISO-8859-1")

    to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
    data = data.drop(columns=to_drop)

    data.rename(
        columns={
            "v1": "Target",
            "v2": "Text",
        },
        inplace=True,
    )

    return data


def build_corpus(data):
    corpus = []

    for text in data["Text"]:
        corpus.append(preprocess(text))

    return corpus


def train_model(x_train, y_train):
    model = MultinomialNB()
    model.fit(x_train, y_train)

    return model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions):.4f}")


def save_artifacts(vectorizer, model, label_encoder):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("\nArtifacts saved:")
    print(f"- {TFIDF_VECTORIZER_PATH}")
    print(f"- {MODEL_PATH}")
    print(f"- {LABEL_ENCODER_PATH}")


def main():
    print("Download NLTK Resources")
    download_nltk_resources()

    print("\nLoad Data")
    data = read_data(DATA_PATH)

    print("Build Corpus")
    corpus = build_corpus(data)

    print("Create Label Encoder")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["Target"])

    print("Split Data")

    text_train, text_test, y_train, y_test = train_test_split(
        corpus,
        y,
        random_state=42,
        test_size=0.2,
    )

    print("Creating TF-IDF Vectorizer")

    vectorizer = TfidfVectorizer()

    x_train = vectorizer.fit_transform(text_train)
    x_test = vectorizer.transform(text_test)

    print("Train Model")

    model = train_model(x_train, y_train)

    print("Evaluate Model")

    evaluate_model(model, x_test, y_test)

    print("\nSaving Artifacts")

    save_artifacts(
        vectorizer,
        model,
        label_encoder,
    )


if __name__ == "__main__":
    main()