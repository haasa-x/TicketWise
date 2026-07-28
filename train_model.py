"""
Requirements covered: #1 (prepare text data), #2 (TF-IDF), #3 (train NB + LogReg),
#4 (compare models, keep the better one).
Run after generate_data.py: python train_model.py
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from utils import clean_text
import config


def load_and_prepare(csv_path=config.DATA_PATH):
    df = pd.read_csv(csv_path)

    # Requirement 1: drop rows with missing subject/description/category
    df = df.dropna(subset=["subject", "description", "category"])
    df = df[(df["subject"].str.strip() != "") & (df["description"].str.strip() != "")]

    # Requirement 1: glue subject + description into one text blob, then clean it
    df["combined_text"] = (df["subject"] + " " + df["description"]).apply(clean_text)
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)  # drop any that became empty
    return df


def train_and_compare(df):
    X = df["combined_text"]
    y = df["category"]

    # Requirement 1: 80% to learn from, 20% held back to quiz the model on later.
    # stratify=y keeps the category mix balanced in both piles.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Requirement 2: turn words into numbers.
    # fit_transform LEARNS the vocabulary from training text only.
    # transform (no fit) reuses that same vocabulary on test text -- never let
    # the test set influence the vectorizer, or the evaluation would be unfair.
    # stop_words="english" removes filler words (the, to, it, is...) so that
    # "top influencing words" later actually shows meaningful terms
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Requirement 3: two different "students" learning the same lesson
    models = {
        "naive_bayes": MultinomialNB(),                    # counts word patterns per category
        "logistic_regression": LogisticRegression(max_iter=1000),  # weighs every word's vote
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)      # learn from the 80% pile
        preds = model.predict(X_test_vec)    # guess on the hidden 20% quiz

        # Requirement 4: score each model on the quiz
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, preds, labels=model.classes_)  # who got confused with whom

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "labels": model.classes_.tolist(),
            "report": classification_report(y_test, preds, zero_division=0),
        }
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    # Requirement 4: whichever model has the higher F1 score wins and gets used later
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best_name}")

    return vectorizer, results, best_name


if __name__ == "__main__":
    df = load_and_prepare()
    vectorizer, results, best_name = train_and_compare(df)
    best_model = results[best_name]["model"]

    # Save the winning model + vectorizer + metrics so app.py can just load
    # them instantly instead of retraining every time someone opens the app
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    metrics_to_save = {
        name: {k: v for k, v in r.items() if k != "model"}
        for name, r in results.items()
    }
    metrics_to_save["best_model"] = best_name
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\nSaved {config.MODEL_PATH}, {config.VECTORIZER_PATH}, {config.METRICS_PATH}")