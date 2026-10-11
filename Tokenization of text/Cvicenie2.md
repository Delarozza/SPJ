# 🤖 Cvičenie 2: Vektorizácia textu (TDM) a Logistická Regresia

> **Oblasť:** Strojové učenie (Machine Learning) & Klasifikácia textu  
> **Úloha:** Detekcia nevyžiadanej pošty (Spam Detection)  
> **Cieľ:** Previesť textové správy na číselnú binárnu term-dokument maticu (TDM), natrénovať model logistickej regresie a vyhodnotiť jeho úspešnosť pomocou kľúčových klasifikačných metrík.

---

## 📐 1. Prevádzanie textu na čísla: Term-Dokument Matica (TDM)

Počítače a algoritmy strojového učenia nerozumejú surovému textu – dáta musia byť reprezentované vo forme číselných vektorov (vektorizácia).

### Čo je Binárna Term-Dokument Matica?
Ide o dvojrozmernú maticu s rozmermi:
$$\text{Počet dokumentov } (N) \times \text{Počet unikátnych termov v slovníku } (M)$$

- **Riadok:** Predstavuje jeden dokument (jeden e-mail / správu).
- **Stĺpec:** Predstavuje jeden unikátny term (slovo) zo slovníka celej kolekcie.
- **Hodnota v bunke ($X_{i, j}$):**
  - **`1`**: ak sa term $j$ nachádza v dokumente $i$.
  - **`0`**: ak sa term $j$ v dokumente $i$ nenachádza.

> ⚠️ **Rozdiel medzi frekvenčnou maticou (BoW) a binárnou maticou:**  
> Štandardný *Bag-of-Words* počíta počet výskytov slova (napr. slovo sa objavilo 3-krát $\rightarrow$ hodnota 3).  
> **Binárna matica** eviduje iba binárnu prítomnosť (presence) $\rightarrow$ iba `1` alebo `0`, bez ohľadu na to, koľkokrát sa slovo opakovalo.

### Prečo použiť riedku maticu (`scipy.sparse.lil_matrix` / `csr_matrix`)?
Väčšina dokumentov obsahuje len desiatky slov z celkového slovníka tisícov termov (99%+ buniek matice obsahuje nuly).  
- Bežná hustá matica (napr. klasické NumPy pole) by zbytočne spotrebovala gigabajty pamäte RAM.
- **Riedka matica (sparse matrix)** ukladá do pamäte iba indexy a hodnoty nenulových prvkov (`1`).
- `lil_matrix` (List of Lists) je ideálna a rýchla na postupné napĺňanie.
- `csr_matrix` (Compressed Sparse Row) je optimálna na rýchle matematické operácie pri trénovaní modelu.

---

## 📈 2. Klasifikácia: Logistická Regresia

**Logistická regresia (Logistic Regression)** je lineárny klasifikačný algoritmus, ktorý odhaduje pravdepodobnosť, že daný vstup patrí do cieľovej triedy (napr. Spam).

- **Cieľový atribút ($y$):** Binárna hodnota  
  - $1$ = **Spam** (pozitívna trieda)  
  - $0$ = **Ham** (bežná, legitímna pošta)
- **Vstupné vlastnosti ($X$):** Riadkový binárny vektor prítomnosti slov správy.

Rozdelenie dát:
- Dátová sada sa delí na **trénovaciu (80%)** a **testovaciu (20%)** množinu.
- Používa sa **stratifikácia (`stratify=y`)**, aby bol pomer tried (spam vs. ham) identický v trénovacej aj testovacej časti (dôležité pri nevyvážených dátach).

---

## 🎯 3. Vyhodnotenie modelu: Klasifikačné metriky

Pri klasifikácii textu (obzvlášť spamu) samotná presnosť nestačí:

| Metrika | Vzorec | Čo znamená v praxi (Detekcia spamu) |
|---|:---:|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Celková percentuálna úspešnosť správnych rozhodnutí. Môže byť skreslená pri nevyváženom datasete. |
| **Precision** | $\frac{TP}{TP + FP}$ | Ak model označil správu ako SPAM, s akou pravdepodobnosťou to **naozaj bol spam**? (Chráni pred označením dôležitého e-mailu za spam). |
| **Recall** | $\frac{TP}{TP + FN}$ | Koľko percent zo všetkých reálnych spamov model **dokázal zachytiť**? |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonický priemer Precision a Recall. Kľúčová metrika pri nerovnomernom rozdelení tried. |

*Vysvetlivky:*  
- **TP (True Positive):** Spam správne označený ako spam.  
- **TN (True Negative):** Bežná pošta správne označená ako bežná.  
- **FP (False Positive):** Bežná pošta chybne označená ako spam (veľký problém pre používateľa!).  
- **FN (False Negative):** Spam, ktorý prekĺzol do doručenej pošty.

---

## 💻 4. Kód riešenia v Pythone

```python
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
    
    # 1 = Spam, 0 = Ham
    X, vocab = build_matrix(df["Message"])
    y = (df["Category"].str.lower() == "spam").astype(int).values

    # Rozdelenie dát
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Trénovanie modelu
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # Vyhodnotenie
    y_pred = model.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()
```

---

## 📌 5. Výsledky nášho modelu
- **Accuracy:** `98.21%`
- **Precision:** `100.00%` (Model neurobil ani jednu chybu typu FP — žiadny legitímny e-mail nebol chybne zahodený ako spam).
- **Recall:** `86.58%` (Model úspešne zachytil ~87% všetkých spamových správ).
- **F1-Score:** `92.81%` (Vynikajúca celková rovnováha klasifikátora).
