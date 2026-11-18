import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

all_rows = []

def add(folder, text_fn, label_fn, source):
    ds = load_from_disk(f"data/raw/{folder}")
    for split in ds:
        for r in ds[split]:
            t = text_fn(r)
            if t and len(t.strip()) > 3:
                all_rows.append({"text": t.strip(), "label": label_fn(r), "source": source})

# --- attacks ---
add("deepset__prompt-injections", lambda r: r["text"], lambda r: int(r["label"]), "deepset")
add("Lakera__gandalf_ignore_instructions", lambda r: r["text"], lambda r: 1, "gandalf")
add("jackhhao__jailbreak-classification", lambda r: r["prompt"],
    lambda r: 1 if r["type"] == "jailbreak" else 0, "jackhhao")

# --- benign (downsampled below) ---
add("databricks__databricks-dolly-15k",
    lambda r: r["instruction"] + (" " + r["context"] if r["context"] else ""),
    lambda r: 0, "dolly")

# SPML EXCLUDED: its 'Prompt injection' column labels the SYSTEM prompt,
# not the user prompt. Training on it teaches the model that innocent
# questions are attacks.

df = pd.DataFrame(all_rows).drop_duplicates(subset="text")

# downsample benign to ~2x the attack count
mal = df[df.label == 1]
ben = df[df.label == 0].sample(n=min(len(mal) * 2, len(df[df.label == 0])), random_state=42)
df = pd.concat([mal, ben]).sample(frac=1, random_state=42).reset_index(drop=True)

print(df.groupby(["source", "label"]).size())
print("TOTAL:", len(df), "| attacks:", (df.label == 1).sum())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2)),
    ("lr", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
])
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred, target_names=["benign", "attack"]))
print(confusion_matrix(y_test, pred))
joblib.dump(clf, "detector.joblib")
print("Saved detector.joblib")
