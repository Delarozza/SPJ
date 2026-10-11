import pandas as pd
import re
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def tokenize_message(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def build_matrix(messages):
    tokenized = [set(tokenize_message(msg)) for msg in messages]
    vocab = sorted(list(set(term for doc in tokenized for term in doc)))
    term_to_idx = {term: idx for idx, term in enumerate(vocab)}

    mat = lil_matrix((len(messages), len(vocab)), dtype=np.uint8)
    for doc_idx, doc_terms in enumerate(tokenized):
        for term in doc_terms:
            if term in term_to_idx:
                mat[doc_idx, term_to_idx[term]] = 1
    return mat.tocsr(), vocab

def main():
    df = pd.read_csv("spam.csv")
    if "v1" in df.columns: 
        df = df.rename(columns={"v1": "Category", "v2": "Message"})
    
    #spam - 1, ham - 0
    X, vocab = build_matrix(df["Message"])
    y = (df["Category"].str.lower() == "spam").astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
