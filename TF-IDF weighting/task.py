import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import lil_matrix, diags, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_message(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def build_frequency_matrix(messages):
    """
    1. Úloha: Frekvenčná term-dokument matica (Term Frequency Matrix).
    Namiesto binárnej hodnoty (0 alebo 1) ukladá skutočný počet výskytov (tf)
    daného termu v príslušnom dokumente.
    """
    tokenized = [tokenize_message(msg) for msg in messages]
    vocab = sorted(list(set(term for doc in tokenized for term in doc)))
    term_to_idx = {term: idx for idx, term in enumerate(vocab)}

    mat = lil_matrix((len(messages), len(vocab)), dtype=np.float64)
    for doc_idx, doc_tokens in enumerate(tokenized):
        counts = Counter(doc_tokens)
        for term, count in counts.items():
            if term in term_to_idx:
                mat[doc_idx, term_to_idx[term]] = count

    return mat.tocsr(), vocab, term_to_idx


def compute_tf(X_freq, scheme="log"):
    """
    Výpočet variantov frekvencie termu (Term Frequency - TF):
      - 'raw': f_{t,d} (priama frekvencia)
      - 'binary': 1 ak f_{t,d} > 0 inak 0
      - 'log': 1 + log10(f_{t,d}) ak f_{t,d} > 0 inak 0
      - 'augmented': 0.5 + 0.5 * (f_{t,d} / max_t(f_{t,d})) ak f_{t,d} > 0 inak 0
      - 'relative': f_{t,d} / sum_t(f_{t,d}) (relatívna frekvencia v dokumente)
    """
    X_tf = X_freq.astype(np.float64, copy=True)
    scheme = scheme.lower()

    if scheme == "raw":
        return X_tf
    elif scheme == "binary":
        X_tf.data = np.ones_like(X_tf.data)
        return X_tf
    elif scheme == "log":
        X_tf.data = 1.0 + np.log10(X_tf.data)
        return X_tf
    elif scheme == "augmented":
        for i in range(X_tf.shape[0]):
            s, e = X_tf.indptr[i], X_tf.indptr[i + 1]
            if s < e:
                m = np.max(X_tf.data[s:e])
                if m > 0:
                    X_tf.data[s:e] = 0.5 + 0.5 * (X_tf.data[s:e] / m)
        return X_tf
    elif scheme == "relative":
        for i in range(X_tf.shape[0]):
            s, e = X_tf.indptr[i], X_tf.indptr[i + 1]
            if s < e:
                tot = np.sum(X_tf.data[s:e])
                if tot > 0:
                    X_tf.data[s:e] = X_tf.data[s:e] / tot
        return X_tf
    else:
        raise ValueError(f"Neznámy variant TF: '{scheme}'. Možnosti: raw, binary, log, augmented, relative.")


def compute_idf(X_freq, scheme="standard"):
    """
    Výpočet variantov inverznej dokumentovej frekvencie (Inverse Document Frequency - IDF):
      - 'none': 1 (bez IDF váhovania)
      - 'standard': log10(N / df)
      - 'smooth': log10(1 + N / df)
      - 'probabilistic': max(0, log10((N - df) / df))
    """
    N = X_freq.shape[0]
    # Document frequency pre každý stĺpec (term)
    df_vals = np.bincount(X_freq.indices, minlength=X_freq.shape[1]).astype(np.float64)
    df_safe = np.maximum(df_vals, 1.0)
    scheme = scheme.lower()

    if scheme == "none":
        return np.ones(X_freq.shape[1], dtype=np.float64)
    elif scheme == "standard":
        return np.log10(N / df_safe)
    elif scheme == "smooth":
        return np.log10(1.0 + N / df_safe)
    elif scheme == "probabilistic":
        prob = np.log10(np.maximum(1.0, N - df_vals) / df_safe)
        return np.maximum(0.0, prob)
    else:
        raise ValueError(f"Neznámy variant IDF: '{scheme}'. Možnosti: none, standard, smooth, probabilistic.")


def compute_tfidf(X_freq, tf_scheme="log", idf_scheme="standard"):
    """
    2. Úloha: Výpočet TF-IDF matice na základe zvolených variantov TF a IDF.
    Výsledná matica: X_tfidf = TF * IDF
    """
    X_tf = compute_tf(X_freq, tf_scheme)
    idf = compute_idf(X_freq, idf_scheme)
    X_tfidf = X_tf.dot(diags(idf)).tocsr()
    return X_tfidf, idf


def vectorize_text(text, vocab, term_to_idx, idf_vector, tf_scheme="log"):
    """
    Prevod textového reťazca (napr. dopytu alebo novej správy) na váhovaný vektor
    so zvoleným variantom TF a daným IDF vektorom.
    """
    tokens = tokenize_message(text)
    counts = Counter(tokens)
    vec = np.zeros(len(vocab), dtype=np.float64)

    for term, count in counts.items():
        if term in term_to_idx:
            idx = term_to_idx[term]
            vec[idx] = count

    # Aplikácia zvoleného TF variantu
    tf_scheme = tf_scheme.lower()
    nonzero_mask = vec > 0

    if tf_scheme == "raw":
        pass
    elif tf_scheme == "binary":
        vec[nonzero_mask] = 1.0
    elif tf_scheme == "log":
        vec[nonzero_mask] = 1.0 + np.log10(vec[nonzero_mask])
    elif tf_scheme == "augmented":
        if np.any(nonzero_mask):
            m = np.max(vec[nonzero_mask])
            vec[nonzero_mask] = 0.5 + 0.5 * (vec[nonzero_mask] / m)
    elif tf_scheme == "relative":
        if np.any(nonzero_mask):
            tot = np.sum(vec[nonzero_mask])
            vec[nonzero_mask] = vec[nonzero_mask] / tot

    # Vynásobenie IDF vektorom
    vec = vec * idf_vector
    return csr_matrix(vec)


def search_query(query, X_tfidf, df, vocab, term_to_idx, idf_vector, tf_scheme="log", top_n=5):
    """
    3. Úloha: Vyhľadávanie informácií pomocou kosínusovej podobnosti.
    Zoradí dokumenty v kolekcii podľa podobnosti s dopytom v zostupnom poradí.
    """
    q_vec = vectorize_text(query, vocab, term_to_idx, idf_vector, tf_scheme=tf_scheme)
    sims = cosine_similarity(q_vec, X_tfidf)[0]
    ranked_indices = np.argsort(sims)[::-1]

    results = []
    for rank, doc_idx in enumerate(ranked_indices[:top_n], start=1):
        results.append({
            "rank": rank,
            "doc_idx": doc_idx,
            "similarity": sims[doc_idx],
            "category": df.iloc[doc_idx]["Category"],
            "message": df.iloc[doc_idx]["Message"]
        })
    return results


def evaluate_logistic_regression(X_tfidf, y, test_size=0.2, random_state=42):
    """
    4. Úloha: Trénovanie Logistickej Regresie a výpočet klasifikačných metrík.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    return model, X_test, y_test, y_pred, metrics


def compare_all_weightings(X_freq, y):
    """
    Porovnanie rôznych kombinácií váhovania TF a IDF.
    """
    schemes = [
        ("binary", "none", "Binárne (Cvičenie 2)"),
        ("raw", "none", "Priama frekvencia (bez IDF)"),
        ("log", "none", "Logaritmické TF (bez IDF)"),
        ("raw", "standard", "Raw TF * Standard IDF"),
        ("log", "standard", "Log TF * Standard IDF (Klasické TF-IDF)"),
        ("log", "smooth", "Log TF * Smooth IDF"),
        ("augmented", "standard", "Augmented TF * Standard IDF"),
        ("raw", "probabilistic", "Raw TF * Probabilistic IDF"),
        ("relative", "standard", "Relative TF * Standard IDF")
    ]

    results = []
    for tf_s, idf_s, desc in schemes:
        X_w, _ = compute_tfidf(X_freq, tf_scheme=tf_s, idf_scheme=idf_s)
        _, _, _, _, metrics = evaluate_logistic_regression(X_w, y)
        results.append({
            "tf": tf_s,
            "idf": idf_s,
            "description": desc,
            **metrics
        })
    return results


def load_dataset():
    """Načítanie datasetu spam.csv s fallbackom na priečinok z predošlého cvičenia."""
    paths = [
        "spam.csv",
        "Cvicenie 2/spam.csv",
        os.path.join(os.path.dirname(__file__), "spam.csv"),
        os.path.join(os.path.dirname(__file__), "..", "Tokenization of text", "spam.csv")
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "v1" in df.columns:
                df = df.rename(columns={"v1": "Category", "v2": "Message"})
            return df
    raise FileNotFoundError("Súbor 'spam.csv' sa nenašiel!")


def main():
    # 1. Načítanie datasetu a frekvenčná matica
    df = load_dataset()
    messages = df["Message"].tolist()
    y = (df["Category"].str.lower() == "spam").astype(int).values

    X_freq, vocab, term_to_idx = build_frequency_matrix(messages)
    sparsity = 100.0 * (1 - X_freq.nnz / (X_freq.shape[0] * X_freq.shape[1]))
    print("=" * 88)
    print(f"Dataset: {len(df)} správ ({np.sum(y == 1)} spam, {np.sum(y == 0)} ham) | "
          f"TDM: {X_freq.shape[0]}x{X_freq.shape[1]} (riedkosť {sparsity:.2f}%)")

    # 2. TF-IDF váhovanie (vybrané: log TF + standard IDF)
    selected_tf = "log"
    selected_idf = "standard"
    X_tfidf, idf_vector = compute_tfidf(X_freq, tf_scheme=selected_tf, idf_scheme=selected_idf)

    # 3. Vyhľadávanie informácií (IR - Kosínová podobnosť)
    query_text = "Free entry to win cash prize or claim award now"
    search_results = search_query(
        query_text, X_tfidf, df, vocab, term_to_idx, idf_vector, tf_scheme=selected_tf, top_n=5
    )
    print(f"\n--- IR Vyhľadávanie: \"{query_text}\" (TF={selected_tf}, IDF={selected_idf}) ---")
    for res in search_results:
        preview = res["message"][:70].replace("\n", " ") + ("..." if len(res["message"]) > 70 else "")
        print(f"  #{res['rank']} [{res['similarity']:.4f}] [{res['category'].upper()}] {preview}")

    # 4. Klasifikácia (Logistická regresia) a porovnanie váhovaní
    model, X_test, y_test, y_pred, base_metrics = evaluate_logistic_regression(X_tfidf, y)
    sample_preds = [f"#{i}:{ 'SPAM' if y_pred[i] == 1 else 'HAM' }({'✓' if y_pred[i] == y_test[i] else '✗'})" for i in range(5)]
    print(f"\n--- Klasifikácia (LogReg) | Ukážka testovacích predikcií: {', '.join(sample_preds)} ---")

    print("\n--- Porovnanie spôsobov váhovania (TF a IDF) ---")
    comparison_table = compare_all_weightings(X_freq, y)
    print(f"{'TF':<11} | {'IDF':<14} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | Popis")
    print("-" * 88)
    for row in comparison_table:
        print(
            f"{row['tf']:<11} | {row['idf']:<14} | {row['accuracy']:<9.4f} | "
            f"{row['precision']:<9.4f} | {row['recall']:<9.4f} | {row['f1']:<9.4f} | {row['description']}"
        )

    # 5. Predikcia nových správ
    print("\n--- Predikcia nových správ ---")
    custom_messages = [
        "URGENT! You have won a 1000 cash prize or a free cruise holiday. Call 0800123456 now to claim!",
        "Ahoj Peter, stretneme sa dnes po obede v univerzitnej knižnici?",
        "Congratulations! Claim your free £500 shopping voucher today by texting WIN to 8888."
    ]

    for msg in custom_messages:
        vec = vectorize_text(msg, vocab, term_to_idx, idf_vector, tf_scheme=selected_tf)
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        label = "SPAM" if pred == 1 else "HAM"
        confidence = proba[1] if pred == 1 else proba[0]
        short_msg = msg if len(msg) <= 65 else msg[:62] + "..."
        print(f"  • \"{short_msg}\" -> {label} ({confidence * 100:.1f}%)")
    print("=" * 88)


if __name__ == "__main__":
    main()
