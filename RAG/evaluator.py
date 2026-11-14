"""
Modul pre vyhodnocovanie a porovnanie troch prístupov:
1. RAG systém (vyhľadané chunky + LLM)
2. Pôvodný document retrieval prístup (priamo nájdený text/chunk)
3. Samostatný jazykový model bez prístupu ku kolekcii (Zero-shot LLM)

Obsahuje štandardné QA metriky (Exact Match, Token F1, Hit@k)
a overenie faktickej vernosti (Faithfulness / Groundedness) odpovede voči kontextu.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import string
from collections import Counter


def normalize_text(s: str) -> str:
    """Normalizácia textu: malé písmená, odstránenie interpunkcie a nadbytočných medzier."""
    s = str(s).lower()
    # Odstránenie interpunkcie
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    # Odstránenie prebytočných medzier
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Vypočíta Exact Match (1.0 ak sa predikcia zhoduje s niektorou z referenčných odpovedí)."""
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return 0.0

    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        if norm_pred == norm_gt or norm_gt in norm_pred:
            return 1.0
    return 0.0


def compute_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Vypočíta štandardné SQuAD token-level F1 skóre voči najlepšej referenčnej odpovedi."""
    norm_pred = normalize_text(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        gt_tokens = norm_gt.split()
        if not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def check_retrieval_hit(ground_truths: List[str], retrieved_texts: List[str]) -> float:
    """Zistí, či sa niektorá správna odpoveď nachádza vo vyhľadaných chunkoch (Recall / Hit@k)."""
    combined_context = normalize_text(" ".join(retrieved_texts))
    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        if norm_gt and norm_gt in combined_context:
            return 1.0
    return 0.0


def check_faithfulness(prediction: str, context: str) -> Tuple[float, str]:
    """
    Overenie, či vygenerovaná odpoveď skutočne vychádza z poskytnutých chunkov (Groundedness / Faithfulness).
    Deteguje halucinácie a overuje správne priznanie nevedomosti.
    """
    # 1. Priznanie nevedomosti
    refusal_keywords = [
        "nie je možné", "nenachádza sa", "kontext neobsahuje", "nie sú informácie",
        "nie je uvedené", "chýbajúce informácie"
    ]
    norm_pred = normalize_text(prediction)
    for kw in refusal_keywords:
        if kw in norm_pred:
            return 1.0, "Správne priznaná absencia informácie v kontexte (Honest Refusal)"

    # 2. Overenie podielu kľúčových slov predikcie prítomných v kontexte
    pred_words = [w for w in norm_pred.split() if len(w) > 2]
    if not pred_words:
        return 1.0, "Prázdna alebo triviálna odpoveď"

    norm_ctx = normalize_text(context)
    ctx_words = set(norm_ctx.split())

    found_words = [w for w in pred_words if w in ctx_words]
    faithfulness_ratio = len(found_words) / len(pred_words)

    if faithfulness_ratio >= 0.70:
        verdict = f"Vysoká vernosť kontextu ({faithfulness_ratio:.1%})"
    elif faithfulness_ratio >= 0.40:
        verdict = f"Čiastočná opora v kontexte ({faithfulness_ratio:.1%})"
    else:
        verdict = f"Potenciálna halucinácia / externá znalosť ({faithfulness_ratio:.1%})"

    return faithfulness_ratio, verdict


class RAGEvaluator:
    """
    Vyhodnocovací modul pre porovnanie RAG, Retrieval-only a Standalone LLM.
    """

    def __init__(self):
        self.results_history = []

    def evaluate_sample(
        self,
        question: str,
        ground_truths: List[str],
        rag_answer: str,
        retrieval_text: str,
        pure_llm_answer: str,
        retrieved_chunks_texts: List[str],
    ) -> Dict[str, Any]:
        """Vyhodnotenie jedného príkladu pre všetky tri prístupy."""
        # 1. Metriky pre RAG
        rag_em = compute_exact_match(rag_answer, ground_truths)
        rag_f1 = compute_f1_score(rag_answer, ground_truths)
        faith_score, faith_verdict = check_faithfulness(rag_answer, " ".join(retrieved_chunks_texts))

        # 2. Metriky pre Retrieval-only
        retr_hit = check_retrieval_hit(ground_truths, retrieved_chunks_texts)
        retr_em = compute_exact_match(retrieval_text, ground_truths)
        retr_f1 = compute_f1_score(retrieval_text, ground_truths)

        # 3. Metriky pre Samostatné LLM
        pure_em = compute_exact_match(pure_llm_answer, ground_truths)
        pure_f1 = compute_f1_score(pure_llm_answer, ground_truths)

        record = {
            "question": question,
            "ground_truths": ground_truths,
            "rag": {
                "answer": rag_answer,
                "em": rag_em,
                "f1": rag_f1,
                "faithfulness_score": faith_score,
                "faithfulness_verdict": faith_verdict,
            },
            "retrieval": {
                "text_snippet": retrieval_text[:200] + "..." if len(retrieval_text) > 200 else retrieval_text,
                "hit": retr_hit,
                "em": retr_em,
                "f1": retr_f1,
            },
            "pure_llm": {
                "answer": pure_llm_answer,
                "em": pure_em,
                "f1": pure_f1,
            }
        }
        self.results_history.append(record)
        return record

    def compute_summary(self) -> Dict[str, Any]:
        """Výpočet celkových priemerných metrík za všetky vyhodnotené vzorky."""
        N = len(self.results_history)
        if N == 0:
            return {}

        avg_rag_em = sum(r["rag"]["em"] for r in self.results_history) / N
        avg_rag_f1 = sum(r["rag"]["f1"] for r in self.results_history) / N
        avg_rag_faith = sum(r["rag"]["faithfulness_score"] for r in self.results_history) / N

        avg_retr_hit = sum(r["retrieval"]["hit"] for r in self.results_history) / N
        avg_retr_f1 = sum(r["retrieval"]["f1"] for r in self.results_history) / N

        avg_pure_em = sum(r["pure_llm"]["em"] for r in self.results_history) / N
        avg_pure_f1 = sum(r["pure_llm"]["f1"] for r in self.results_history) / N

        return {
            "total_questions": N,
            "rag": {
                "avg_exact_match": avg_rag_em,
                "avg_f1": avg_rag_f1,
                "avg_faithfulness": avg_rag_faith,
            },
            "retrieval": {
                "hit_rate_at_k": avg_retr_hit,
                "avg_f1": avg_retr_f1,
            },
            "pure_llm": {
                "avg_exact_match": avg_pure_em,
                "avg_f1": avg_pure_f1,
            }
        }
