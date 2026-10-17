# 📊 Lab 3: Frequency TDM, TF-IDF Weighting, Information Retrieval (IR) and Classification

> **Domain:** Natural Language Processing (NLP) & Information Retrieval (IR)  
> **Task:** SMS Spam Detection and Semantic Information Retrieval  
> **Objective:** Transition from a binary term-document matrix to a frequency-based matrix, implement various Term Frequency (TF) and Inverse Document Frequency (IDF) schemes, conduct query-based retrieval via cosine similarity, train a Logistic Regression classifier, and evaluate the performance of different weighting combinations.

---

## 📐 1. Frequency Term-Document Matrix (Frequency TDM)

Unlike a binary matrix (where each cell contains only `0` or `1` indicating word presence), a **frequency term-document matrix** records the exact count of occurrences (term frequency – $f_{t, d}$) of each word $t$ in document $d$:

$$X_{d, t} = f_{t, d}$$

- **Rows ($N$):** Documents / messages in the dataset ($N = 5572$).
- **Columns ($M$):** Unique terms across the entire corpus vocabulary ($M = 8749$).
- **Sparsity:** The matrix contains $81\,883$ non-zero elements ($99.83\%$ of elements are zeros). It is stored using compressed sparse row format (`scipy.sparse.csr_matrix`), significantly reducing memory consumption and accelerating vector operations.

---

## 🧮 2. TF-IDF Weighting and Implemented Variants

The TF-IDF weight assigns importance to term $t$ within document $d$ as the product of its local document frequency (**TF**) and its global rarity across the corpus (**IDF**):

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

### A. Term Frequency (TF) Variants:

1. **Raw Frequency:**
   $$\text{TF}_{\text{raw}}(t, d) = f_{t, d}$$

2. **Binary:**
   $$\text{TF}_{\text{bin}}(t, d) = \begin{cases} 1 & \text{if } f_{t, d} > 0 \\ 0 & \text{otherwise} \end{cases}$$

3. **Logarithmic:**
   Sublinearly dampens high-frequency terms so that a term appearing 10 times is not weighted 10 times higher:
   $$\text{TF}_{\text{log}}(t, d) = \begin{cases} 1 + \log_{10}(f_{t, d}) & \text{if } f_{t, d} > 0 \\ 0 & \text{otherwise} \end{cases}$$

4. **Augmented (Double Normalization 0.5):**
   Prevents bias towards longer documents by scaling with respect to the maximum term frequency within the document:
   $$\text{TF}_{\text{aug}}(t, d) = \begin{cases} 0.5 + 0.5 \cdot \frac{f_{t, d}}{\max_{t'} f_{t', d}} & \text{if } f_{t, d} > 0 \\ 0 & \text{otherwise} \end{cases}$$

5. **Relative Frequency:**
   Length normalization dividing by the total word count of the document:
   $$\text{TF}_{\text{rel}}(t, d) = \frac{f_{t, d}}{\sum_{t'} f_{t', d}}$$

### B. Inverse Document Frequency (IDF) Variants:

Let $N$ denote the total number of documents in the collection, and $df_t$ the document frequency (number of documents containing term $t$):

1. **Unary / None:**
   $$\text{IDF}_{\text{none}}(t) = 1$$

2. **Standard IDF:**
   $$\text{IDF}_{\text{standard}}(t) = \log_{10}\left(\frac{N}{df_t}\right)$$

3. **Smooth IDF:**
   Mitigates edge cases and smooths the distribution:
   $$\text{IDF}_{\text{smooth}}(t) = \log_{10}\left(1 + \frac{N}{df_t}\right)$$

4. **Probabilistic IDF:**
   Derived from odds ratio in probabilistic retrieval models:
   $$\text{IDF}_{\text{prob}}(t) = \max\left(0, \log_{10}\left(\frac{N - df_t}{df_t}\right)\right)$$

---

## 🔍 3. Information Retrieval (IR) via Cosine Similarity

For a given text query $q$ (e.g., *"Free entry to win cash prize or claim award now"*):
1. The query string is tokenized identically to the corpus documents.
2. A term count vector is formed and weighted using the selected TF scheme and the corpus IDF vector: $\vec{q} = \text{TF}(q) \odot \vec{\text{IDF}}$.
3. **Cosine Similarity** is calculated between query vector $\vec{q}$ and each document vector $\vec{d_i}$:

$$\text{sim}(\vec{q}, \vec{d_i}) = \frac{\vec{q} \cdot \vec{d_i}}{\|\vec{q}\|_2 \cdot \|\vec{d_i}\|_2}$$

4. Documents are sorted in descending order of similarity.

### Top 5 Search Results for the Sample Query:

| Rank | Cosine Similarity | Category | Document Text (Excerpt) |
|:---:|:---:|:---:|---|
| **#1** | **0.4171** | `SPAM` | Win a £1000 cash prize or a prize worth £5000 |
| **#2** | **0.3247** | `SPAM` | You have won ?1,000 cash or a ?2,000 prize! To claim, call09050000327 |
| **#3** | **0.3045** | `SPAM` | You have won a guaranteed 32000 award or maybe even £1000 cash to claim ur... |
| **#4** | **0.2913** | `SPAM` | You have WON a guaranteed £1000 cash or a £2000 prize.To claim yr prize call... |
| **#5** | **0.2516** | `SPAM` | You are a winner you have been specially selected to receive £1000 cash... |

---

## 📈 4. Classification using Logistic Regression

The weighted TF-IDF matrix is split into **80% training** and **20% test** subsets using stratified sampling (`stratify=y`, `random_state=42`).  
A **Logistic Regression** model is trained to predict the target class (`1 = SPAM`, `0 = HAM`).

### Performance Comparison Across Weighting Schemes:

| TF Variant | IDF Variant | Accuracy | Precision | Recall | F1-Score | Remarks |
|---|---|:---:|:---:|:---:|:---:|---|
| **binary** | **none** | 98.21% | 100.00% | 86.58% | **92.81%** | Baseline from Lab 2 |
| **raw** | **none** | 98.57% | 99.26% | 89.93% | **94.37%** | Raw count without IDF |
| **log** | **none** | 98.30% | 100.00% | 87.25% | **93.19%** | Log TF dampens extreme counts |
| **raw** | **standard** | 98.57% | 100.00% | 89.26% | **94.33%** | Raw TF * Standard IDF |
| **log** | **standard** | **98.57%** | **100.00%** | **89.26%** | **94.33%** | Classic TF-IDF weighting |
| **log** | **smooth** | 98.48% | 100.00% | 88.59% | **93.95%** | Smoothed IDF |
| **augmented** | **standard** | 98.30% | 100.00% | 87.25% | **93.19%** | Max-TF normalized |
| **raw** | **probabilistic** | **98.65%** | **100.00%** | **89.93%** | **94.70%** | **Highest F1-Score achieved** |
| **relative** | **standard** | 88.97% | 100.00% | 17.45% | **29.71%** | Significant recall drop due to short document lengths |

---

## 🔮 5. Prediction on New Unseen Messages

Each custom message is converted into a weighted feature vector and evaluated by the trained Logistic Regression model:

1. **Input:** `"URGENT! You have won a 1000 cash prize or a free cruise holiday. Call 0800123456 now to claim!"`  
   - **Prediction:** `SPAM`  
   - **Confidence:** HAM: 0.11%, SPAM: **99.89%**

2. **Input:** `"Ahoj Peter, stretneme sa dnes po obede v univerzitnej knižnici?"`  
   - **Prediction:** `HAM`  
   - **Confidence:** HAM: **99.44%**, SPAM: 0.56%

3. **Input:** `"Congratulations! Claim your free £500 shopping voucher today by texting WIN to 8888."`  
   - **Prediction:** `SPAM`  
   - **Confidence:** HAM: 17.29%, SPAM: **82.71%**

---

## 💡 6. Key Takeaways & Conclusions

1. **Transition from Binary to Frequency TF-IDF:** Incorporating term frequencies and inverse document frequencies improved model performance from an F1-score of **92.81%** (binary matrix) to **94.70%** (Raw TF + Probabilistic IDF), while maintaining **100.00% Precision** (zero false positives).
2. **Cosine Similarity in IR:** Computes semantic relevance effectively without heavy computational overhead, successfully ranking spam promotion messages at the top for monetary/prize queries.
3. **Weighting Scheme Impact:** Suppressing common stop words while emphasizing discriminative keywords (`urgent`, `prize`, `claim`, `win`) provides stronger feature separation for both classification and retrieval.
