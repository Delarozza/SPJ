#!/usr/bin/env python3
"""
Hlavný riadiaci skript pre Cvičenie 7 - RAG (Retrieval-Augmented Generation).
SPj (Spracovanie prirodzeného jazyka), TUKE / FEI.

Spája všetky moduly:
1. Rozdelenie kolekcie na chunky (chunker.py)
2. Vytvorenie embeddingov (embeddings.py)
3. Vyhľadávanie relevantných chunkov (retriever.py)
4. RAG pipeline s LLM < 10B parametrov (rag_pipeline.py)
5. Porovnanie troch prístupov a vyhodnotenie (evaluator.py)
"""

import os
import logging
import sys
import argparse
import json
import time
import importlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Lokálne moduly
from chunker import TextChunker, TextChunk
from embeddings import EmbeddingEngine
from retriever import ChunkRetriever, SparseTfidfRetriever, RetrievalResult
from rag_pipeline import RAGPipeline, DEFAULT_SYSTEM_PROMPT
from evaluator import RAGEvaluator


logger = logging.getLogger(__name__)


def load_skquad_data(split: str = "validation", max_samples: int = 50) -> List[Dict[str, Any]]:
    """Načítanie otázok a referenčných odpovedí z datasetu TUKE-DeutscheTelekom/skquad."""
    logger.info(f"\n[Dáta] Načítavam otázky z 'TUKE-DeutscheTelekom/skquad' (split={split})...")
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Chýba závislosť 'datasets'; nainštalujte ju pomocou: pip install datasets") from exc
    dataset = load_dataset("TUKE-DeutscheTelekom/skquad", split=split)

    valid_samples = []
    for item in dataset:
        answers = item.get("answers", {}).get("text", [])
        question = item.get("question", "").strip()
        context = item.get("context", "").strip()
        title = item.get("title", "").strip()
        if question and answers and context:
            valid_samples.append({
                "id": str(item.get("id")),
                "title": title,
                "question": question,
                "context": context,
                "answers": answers,
            })
            if len(valid_samples) >= max_samples:
                break

    logger.info(f"[Dáta] Načítaných {len(valid_samples)} validných otázok s referenčnými odpoveďami.")
    return valid_samples


def load_skwiki_documents(
    sample_size: int = 500,
    target_docs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Načítanie vzorky dokumentov z datasetu PiotrSty/skwiki-slovak.
    Zahŕňa zdrojové dokumenty pre testovacie otázky plus náhodné články do počtu sample_size.
    """
    logger.info(f"\n[Dáta] Načítavam kolekciu dokumentov (cieľová veľkosť: {sample_size} dokumentov)...")
    from datasets import load_dataset # type: ignore

    documents = []
    seen_ids = set()

    # 1. Pridanie zdrojových článkov zo skquad
    if target_docs:
        for t in target_docs:
            doc_id = f"skquad_doc_{t['id']}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                documents.append({
                    "id": doc_id,
                    "title": t["title"],
                    "text": t["context"],
                    "source": "skquad_reference",
                })

    # 2. Pridanie článkov zo skwiki-slovak ako rozptylových dokumentov (distractors)
    needed = max(0, sample_size - len(documents))
    if needed > 0:
        logger.info(f"[Dáta] Doplňujem {needed} článkov zo slovenskej Wikipédie (PiotrSty/skwiki-slovak)...")
        skwiki = load_dataset("PiotrSty/skwiki-slovak", split="train", streaming=True)
        count = 0
        for item in skwiki:
            doc_id = str(item.get("id", f"wiki_{count}"))
            text = str(item.get("text", "")).strip()
            # Filtrujeme extrémne krátke alebo prázdne články
            if doc_id not in seen_ids and len(text) > 150:
                seen_ids.add(doc_id)
                # Získame prvý riadok ako orientačný titulok
                first_line = text.split("\n")[0][:80].strip()
                title = first_line if first_line else f"Článok {doc_id}"
                documents.append({
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "source": "skwiki-slovak",
                })
                count += 1
                if count >= needed:
                    break

    logger.info(f"[Dáta] Celkový počet dokumentov v indexovanej kolekcii: {len(documents)}")
    return documents


def run_pipeline(args):
    """Hlavný beh programu."""
    start_time = time.time()
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    cache_embeddings_file = os.path.join(
        cache_dir, f"embeddings_s{args.sample_size}_cs{args.chunk_size}_co{args.chunk_overlap}.npz"
    )
    cache_chunks_file = os.path.join(
        cache_dir, f"chunks_s{args.sample_size}_cs{args.chunk_size}_co{args.chunk_overlap}.json"
    )

    # -------------------------------------------------------------------------
    # KROK 1: Načítanie dát
    # -------------------------------------------------------------------------
    print(f"RAG  {args.model_name}  načítavam dáta a modely…", flush=True)
    eval_data = load_skquad_data(split="validation", max_samples=args.num_eval_questions)

    # -------------------------------------------------------------------------
    # KROK 2 & 3: Chunkovanie a Embeddingy (s cachovaním)
    # -------------------------------------------------------------------------
    chunks: List[TextChunk] = []
    embeddings = np.empty((0, 0), dtype=np.float32)

    embedding_engine = EmbeddingEngine(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
    )

    if not args.no_cache and os.path.exists(cache_embeddings_file) and os.path.exists(cache_chunks_file):
        logger.info(f"\n[Cache] Načítavam chunky a embeddingy z lokálneho disku...")
        with open(cache_chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            chunks = [TextChunk.from_dict(d) for d in chunks_data]
        embeddings, _ = EmbeddingEngine.load_cache(cache_embeddings_file)
        logger.info(f"[Cache] Úspešne načítaných {len(chunks)} chunkov a {embeddings.shape[0]} vektorov.")
    else:
        documents = load_skwiki_documents(sample_size=args.sample_size, target_docs=eval_data)

        logger.info(f"\n--- 1. Rozdelenie kolekcie na chunky ---")
        logger.info(f"Parametre: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, jednotka={args.unit}")
        chunker = TextChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            unit=args.unit,
        )
        chunks = chunker.chunk_collection(documents)
        logger.info(f"Celkový počet vytvorených chunkov: {len(chunks)}")
        if chunks:
            sizes = [c.size for c in chunks]
            logger.info(f"Priemerná veľkosť chunku: {np.mean(sizes):.1f} {args.unit} (min: {min(sizes)}, max: {max(sizes)})")

        logger.info(f"\n--- 2. Tvorba vektorových reprezentácií (embeddingov) ---")
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_engine.embed_texts(chunk_texts, show_progress=args.verbose)
        logger.info(f"Rozmer matice embeddingov: {embeddings.shape} (d={embeddings.shape[1]})")

        # Uloženie do cache
        with open(cache_chunks_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
        EmbeddingEngine.save_cache(
            cache_embeddings_file, embeddings, [c.to_dict() for c in chunks]
        )

    # -------------------------------------------------------------------------
    # KROK 4: Inicializácia vyhľadávačov (Dense Retriever & Sparse TF-IDF)
    # -------------------------------------------------------------------------
    retriever = ChunkRetriever(chunks=chunks, embeddings=embeddings, engine=embedding_engine)

    logger.info("\nInicializujem porovnávací TF-IDF retriever...")
    chunks_raw = [{"id": c.chunk_id, "text": c.text, "title": c.title, "doc_id": c.doc_id} for c in chunks]
    tfidf_retriever = SparseTfidfRetriever(chunks_raw, text_key="text")

    # -------------------------------------------------------------------------
    # KROK 5: Inicializácia RAG Pipeline (LLM < 10B)
    # -------------------------------------------------------------------------
    logger.info(f"\n--- 3. Inicializácia generatívneho jazykového modelu ---")
    rag = RAGPipeline(
        model_name=args.model_name,
        backend=args.backend,
        ollama_url=args.ollama_url,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    logger.info(f"Použitý model: {args.model_name}")
    logger.info(f"Odhadovaná veľkosť: {getattr(rag, 'param_count_str', '< 10B')}")
    logger.info("Systémový prompt:")
    logger.info("  " + DEFAULT_SYSTEM_PROMPT.replace("\n", "\n  "))

    # -------------------------------------------------------------------------
    # KROK 6: Porovnanie troch prístupov a vyhodnotenie odpovedí
    # -------------------------------------------------------------------------
    logger.info(f"\n" + "=" * 80)
    logger.info(f" 4. POROVNANIE PRÍSTUPOV A VYHODNOTENIE ODPOVEDÍ (Počet otázok: {len(eval_data)})")
    logger.info("=" * 80)

    evaluator = RAGEvaluator()

    for idx, sample in enumerate(eval_data, 1):
        if not args.verbose:
            print(f"Vyhodnotenie {idx}/{len(eval_data)}…", flush=True)
        q = sample["question"]
        gts = sample["answers"]
        logger.info(f"\n{'='*80}")
        logger.info(f"Otázka #{idx}: {q}")
        logger.info(f"Referenčné odpovede (Ground Truth): {gts}")
        logger.info(f"{'='*80}")

        # 1. Retrieval (Top-k relevantných chunkov)
        retrieved_results = retriever.retrieve(query=q, top_k=args.top_k)
        if args.verbose:
            retriever.print_results(query=q, results=retrieved_results, max_chars=220)

        # 2. Pôvodný retrieval výstup (Top-1 nájdený chunk ako odpoveď)
        retrieval_top_text = retrieved_results[0].chunk.text if retrieved_results else ""

        # 3. RAG výstup (LLM + kontext nájdených chunkov)
        rag_res = rag.answer_with_rag(question=q, retrieved_results=retrieved_results)
        rag_answer = rag_res["answer"]

        # 4. Samostatné LLM (Zero-shot bez prístupu ku kolekcii)
        pure_res = rag.answer_pure_llm(question=q)
        pure_answer = pure_res["answer"]

        # 5. Vyhodnotenie
        eval_record = evaluator.evaluate_sample(
            question=q,
            ground_truths=gts,
            rag_answer=rag_answer,
            retrieval_text=retrieval_top_text,
            pure_llm_answer=pure_answer,
            retrieved_chunks_texts=[r.chunk.text for r in retrieved_results],
        )

        logger.info("\n--- POROVNANIE ODPOVEDÍ ---")
        logger.info(f"• [1. RAG Odpoveď]:     {rag_answer}")
        logger.info(f"    -> EM: {eval_record['rag']['em']:.1f} | F1: {eval_record['rag']['f1']:.4f} | Faithfulness: {eval_record['rag']['faithfulness_verdict']}")
        logger.info(f"• [2. Retrieval-only]:  {retrieval_top_text[:140]}...")
        logger.info(f"    -> Hit@{args.top_k}: {eval_record['retrieval']['hit']:.1f} | Token F1: {eval_record['retrieval']['f1']:.4f}")
        logger.info(f"• [3. Samostatné LLM]:  {pure_answer}")
        logger.info(f"    -> EM: {eval_record['pure_llm']['em']:.1f} | F1: {eval_record['pure_llm']['f1']:.4f}")

    # -------------------------------------------------------------------------
    # KROK 7: Test na otázku bez odpovede v kontexte (Honest Refusal Test)
    # -------------------------------------------------------------------------
    logger.info(f"\n" + "=" * 80)
    logger.info(" TEST ODMIETNUTIA PRI CHÝBAJÚCOM KONTEXTE (Honest Refusal / Zákaz Halucinácií)")
    logger.info("=" * 80)
    print("Test otázky bez odpovede…", flush=True)
    unanswerable_q = "Koľko obyvateľov malo mesto Atlantis v roku 1492 pred naším letopočtom?"
    logger.info(f"Otázka: {unanswerable_q}")
    unans_retrieved = retriever.retrieve(query=unanswerable_q, top_k=2)
    unans_rag = rag.answer_with_rag(question=unanswerable_q, retrieved_results=unans_retrieved)
    unans_pure = rag.answer_pure_llm(question=unanswerable_q)
    logger.info(f"• RAG reakcia:         {unans_rag['answer']}")
    logger.info(f"• Samostatné LLM:     {unans_pure['answer']}")

    # -------------------------------------------------------------------------
    # KROK 8: Súhrnné výsledky a štatistika
    # -------------------------------------------------------------------------
    summary = evaluator.compute_summary()
    print("\nVýsledky:")
    header = f"{'Prístup':<25} | {'Exact Match (EM)':<18} | {'Priemerné F1':<15} | {'Dodatočná metrika':<25}"
    print(header)
    print(f"{'1. RAG Systém':<25} | {summary['rag']['avg_exact_match']:<18.2%} | {summary['rag']['avg_f1']:<15.4f} | Faithfulness: {summary['rag']['avg_faithfulness']:.1%}")
    print(f"{'2. Samotný Retrieval':<25} | {'N/A (pasívny text)':<18} | {summary['retrieval']['avg_f1']:<15.4f} | Hit@{args.top_k}: {summary['retrieval']['hit_rate_at_k']:.1%}")
    print(f"{'3. Samostatné LLM':<25} | {summary['pure_llm']['avg_exact_match']:<18.2%} | {summary['pure_llm']['avg_f1']:<15.4f} | Bez prístupu ku kontextu")

    # Uloženie výsledkov
    results_path = os.path.join(cache_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "summary": summary,
            "samples": evaluator.results_history,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nVýsledky boli úspešne uložené do: {results_path}")

    # -------------------------------------------------------------------------
    # Interaktívny režim
    # -------------------------------------------------------------------------
    if args.interactive:
        print("\n" + "=" * 80)
        print(" INTERAKTÍVNY REŽIM (Zadajte otázku v slovenčine, alebo 'exit' pre ukončenie)")
        print("=" * 80)
        while True:
            try:
                user_q = input("\nVaša otázka > ").strip()
                if not user_q or user_q.lower() in ["exit", "quit", "q"]:
                    break
                user_topk = retriever.retrieve(user_q, top_k=args.top_k)
                if args.verbose:
                    retriever.print_results(user_q, user_topk)
                user_ans = rag.answer_with_rag(user_q, user_topk)
                print(f"\n[RAG Odpoveď]:\n{user_ans['answer']}\n")
            except (KeyboardInterrupt, EOFError):
                break

    print(f"\nCelkový čas behu: {time.time() - start_time:.1f} sekúnd.")


def parse_args():
    parser = argparse.ArgumentParser(description="Cvičenie 7 - RAG Pipeline pre slovenčinu")
    parser.add_argument("--chunk-size", type=int, default=150, help="Veľkosť chunku v slovách (predvolené: 150)")
    parser.add_argument("--chunk-overlap", type=int, default=30, help="Prekrytie chunkov v slovách (predvolené: 30)")
    parser.add_argument("--unit", type=str, default="words", choices=["words", "tokens"], help="Jednotka veľkosti chunku")
    parser.add_argument("--top-k", type=int, default=3, help="Počet vyhľadaných chunkov pre kontext (predvolené: 3)")
    parser.add_argument("--sample-size", type=int, default=300, help="Počet dokumentov v kolekcii (predvolené: 300)")
    parser.add_argument("--num-eval-questions", type=int, default=10, help="Počet otázok zo skquad na vyhodnotenie (predvolené: 10)")
    parser.add_argument("--model-name", type=str, default="qwen2.5:1.5b", help="Názov modelu (<10B, napr. qwen2.5:1.5b pre Ollama alebo Qwen/Qwen2.5-1.5B-Instruct pre Transformers)")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "transformers"], help="Backend pre LLM (ollama alebo transformers)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL pre Ollama")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Viacjazyčný embedding model")
    parser.add_argument("--batch-size", type=int, default=32, help="Dávka pre embeddingy")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximálny počet vygenerovaných tokenov")
    parser.add_argument("--temperature", type=float, default=0.1, help="Teplota pre generovanie")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Priečinok pre cache dát a embeddingov")
    parser.add_argument("--no-cache", action="store_true", help="Ignorovať cache a prepočítať všetko")
    parser.add_argument("--verbose", "-v", action="store_true", help="Podrobný výpis vrátane kontextu a odpovedí")
    parser.add_argument("--interactive", action="store_true", help="Spustiť interaktívny režim pre otázky")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in (__name__, "embeddings", "rag_pipeline"):
        logging.getLogger(name).setLevel(logging.INFO if args.verbose else logging.WARNING)
    if not args.verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("DATASETS_DISABLE_PROGRESS_BARS", "1")
    run_pipeline(args)
