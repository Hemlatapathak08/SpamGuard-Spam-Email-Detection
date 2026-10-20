
from pathlib import Path
import email
from email import policy
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def extract_text(message):
    """Extract subject and readable text from an email."""
    parts = [message.get("subject", "") or ""]

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue

            if part.get_content_disposition() == "attachment":
                continue

            if part.get_content_type() == "text/plain":
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        parts.append(content)
                except Exception:
                    continue
    else:
        if message.get_content_type() == "text/plain":
            try:
                content = message.get_content()
                if isinstance(content, str):
                    parts.append(content)
            except Exception:
                pass

    return "\n".join(parts).strip()


def load_emails(folder, label):
    emails = []

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return emails

    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            raw = file_path.read_bytes()
            msg = email.message_from_bytes(
                raw, policy=policy.default
            )
            text = extract_text(msg)

            if len(text.strip()) >= 10:
                emails.append({
                    "label": label,
                    "message": text
                })
        except Exception:
            continue

    return emails


# Load actual email messages
records = []
records.extend(load_emails(DATA_DIR / "easy_ham", "ham"))
records.extend(load_emails(DATA_DIR / "spam", "spam"))

data = pd.DataFrame(records)

if data.empty:
    raise ValueError(
        "No emails found. Check data/easy_ham and data/spam folders."
    )

print("\nDataset loaded successfully")
print("Total emails:", len(data))
print("\nClass distribution:")
print(data["label"].value_counts())

if data["label"].nunique() < 2:
    raise ValueError(
        "Both easy_ham and spam folders must contain valid emails."
    )

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
   ("classifier", MultinomialNB(alpha=0.03, fit_prior=False))
])

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", f"{accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    zero_division=0
))

print("Confusion Matrix (ham, spam):")
print(confusion_matrix(
    y_test, predictions, labels=["ham", "spam"]
))

# Save trained model
with open(BASE_DIR / "spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save held-out evaluation results for the dashboard
results = {
    "accuracy": accuracy_score(y_test, predictions),
    "report": classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        output_dict=True,
        zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ).tolist(),
    "test_size": len(y_test),
    "train_size": len(y_train),
    "labels": ["ham", "spam"]
}

with open(BASE_DIR / "evaluation.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nModel saved: spam_model.pkl")
print("Evaluation saved: evaluation.pkl")

print("\n--- Spam Prediction Test ---")

test_messages = [
    "Congratulations! You have won a free prize. Click now to claim your reward!",
    "Hi, are we meeting at the college library tomorrow to complete our assignment?",
    "WIN a free ticket now! Call immediately to claim your exclusive cash reward."
]

for msg in test_messages:
    result = model.predict([msg])[0]
    print(f"\nMessage: {msg}")
    print(f"Prediction: {result}")