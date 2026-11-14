"""
Modul pre vyhľadávanie relevantných chunkov (Retrieval).
Implementuje sémantické vektorové vyhľadávanie (Dense Retrieval)
aj frekvenčné TF-IDF vyhľadávanie (Sparse Retrieval) prispôsobené pre chunky.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from collections import Counter
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from chunker import TextChunk
from embeddings import EmbeddingEngine


@dataclass
class RetrievalResult:
    chunk: TextChunk
    score: float
    rank: int


class ChunkRetriever:
    """
    Vyhľadávač relevantných chunkov na základe kosínusovej podobnosti embeddingov (Dense Retrieval).
    """

    def __init__(self, chunks: List[TextChunk], embeddings: np.ndarray, engine: EmbeddingEngine):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Počet chunkov ({len(chunks)}) sa nezhoduje s počtom embeddingov ({embeddings.shape[0]})")

        self.chunks = chunks
        self.embeddings = embeddings  # predpokladáme L2-normalizované
        self.engine = engine

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Vyhľadá top-k najpodobnejších chunkov pre zadaný dopyt/otázku.
        Využíva kosínovú podobnosť (skalárny súčin normalizovaných vektorov).
        """
        if not self.chunks:
            return []

        # Získanie embeddingu otázky
        q_emb = self.engine.embed_query(query)  # tvar (hidden_dim,)

        # Kosínová podobnosť: q_emb @ embeddings.T
        scores = np.dot(self.embeddings, q_emb)

        # Výber top-k indexov
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results

    def print_results(self, query: str, results: List[RetrievalResult], max_chars: int = 300) -> None:
        """Formátované zobrazenie nájdených chunkov v konzole."""
        print(f"\n{'='*80}")
        print(f"DOPYT: \"{query}\"")
        print(f"{'='*80}")
        if not results:
            print("Neboli nájdené žiadne relevantné chunky.")
            return

        for res in results:
            ch = res.chunk
            print(f"\n[Poradie #{res.rank}] | Skóre podobnosti: {res.score:.4f}")
            print(f"Zdrojový dokument: ID={ch.doc_id} | Titulok: \"{ch.title}\" | Chunk #{ch.chunk_index} ({ch.size} {ch.unit})")
            print(f"Rozsah znakov: {ch.start_char}..{ch.end_char}")
            print("-" * 80)
            preview = ch.text if len(ch.text) <= max_chars else ch.text[:max_chars] + "..."
            print(preview)
            print("-" * 80)


class SparseTfidfRetriever:
    """
    Pôvodný frekvenčný TF-IDF vyhľadávač (z Cvičenia 3),
    adaptovaný na vyhľadávanie v chunkoch alebo celých dokumentoch.
    """

    def __init__(self, items: List[Dict[str, Any]], text_key: str = "text"):
        self.items = items
        self.text_key = text_key
        self.vocab, self.term_to_idx, self.X_tfidf = self._build_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", str(text).lower())

    def _build_index(self):
        tokenized_corpus = [self._tokenize(item[self.text_key]) for item in self.items]
        vocab = sorted(list(set(w for doc in tokenized_corpus for w in doc)))
        term_to_idx = {w: i for i, w in enumerate(vocab)}

        N = len(self.items)
        M = len(vocab)
        if N == 0 or M == 0:
            return vocab, term_to_idx, csr_matrix((0, 0))

        # Frekvenčná matica
        freq_mat = lil_matrix((N, M), dtype=np.float64)
        df = Counter()
        for i, tokens in enumerate(tokenized_corpus):
            counts = Counter(tokens)
            for w, c in counts.items():
                if w in term_to_idx:
                    freq_mat[i, term_to_idx[w]] = c
            df.update(counts.keys())

        freq_mat = freq_mat.tocsr()

        # Logaritmické TF: 1 + log10(tf)
        tf_mat = freq_mat.copy().astype(np.float64)
        tf_mat.data = 1.0 + np.log10(tf_mat.data)

        # Štandardné IDF: log10(N / df)
        idf_vec = np.zeros(M, dtype=np.float64)
        for w, idx in term_to_idx.items():
            doc_freq = df.get(w, 1)
            idf_vec[idx] = np.log10((N + 1) / (doc_freq + 1)) + 1.0

        # TF-IDF = TF * IDF
        from scipy.sparse import diags
        tfidf_mat = tf_mat.dot(diags(idf_vec))
        return vocab, term_to_idx, tfidf_mat.tocsr()

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Vyhľadávanie dopytu cez TF-IDF a kosínovú podobnosť."""
        q_tokens = self._tokenize(query)
        tfidf_matrix = self.X_tfidf
        if not q_tokens or tfidf_matrix is None or not self.items or not self.vocab:
            return []

        q_counts = Counter(q_tokens)
        q_vec = lil_matrix((1, len(self.vocab)), dtype=np.float64)
        for w, c in q_counts.items():
            if w in self.term_to_idx:
                q_vec[0, self.term_to_idx[w]] = 1.0 + np.log10(c)

        q_vec = q_vec.tocsr()
        scores = cosine_similarity(csr_matrix(q_vec), csr_matrix(tfidf_matrix))[0]

        top_k = min(top_k, len(self.items))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.items[i], float(scores[i])) for i in top_indices]
