"""
Modul pre tvorbu a cachovanie vektorových reprezentácií (embeddingov) textov a chunkov.
Používa osvedčené viacjazyčné modely (napr. paraphrase-multilingual-MiniLM-L12-v2)
s mean-poolingom a L2 normalizáciou pre bleskovú kosínovú podobnosť cez skalárny súčin.
"""

import os
import logging
from importlib import import_module
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Generátor textových embeddingov využívajúci HuggingFace Transformers.
    Automaticky využíva GPU (CUDA), ak je k dispozícii, inak beží na CPU.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        try:
            self.torch = import_module("torch")
        except ImportError as exc:
            raise ImportError(
                "EmbeddingEngine requires PyTorch. Install it with: pip install torch"
            ) from exc

        if device is None:
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[EmbeddingEngine] Načítavam model: {self.model_name} na zariadenie: {self.device}")
        try:
            transformers = import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "EmbeddingEngine requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from exc

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.model = transformers.AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """Výpočet priemeru tokenových reprezentácií s maskovaním paddingu."""
        token_embeddings = model_output[0]  # [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = self.torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = self.torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Vygeneruje L2-normalizované embeddingy pre zoznam textov.
        Vráti NumPy maticu s tvarom (len(texts), hidden_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        all_embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress and len(texts) > self.batch_size:
            iterator = tqdm(iterator, desc="Generovanie embeddingov", leave=False)

        with self.torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)
                pooled = self._mean_pooling(outputs, encoded["attention_mask"])
                # L2 normalizácia (vektory s jednotkovou normou)
                normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Vygeneruje L2-normalizovaný vektor pre jednu otázku/dopyt."""
        return self.embed_texts([query], show_progress=False)[0]

    @staticmethod
    def save_cache(cache_path: str, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Uloženie embeddingov a metadát chunkov do komprimovaného .npz súboru."""
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            metadata=np.array(metadata, dtype=object),
        )
        logger.info(f"[EmbeddingEngine] Cache úspešne uložená do {cache_path} ({embeddings.shape[0]} vektorov)")

    @staticmethod
    def load_cache(cache_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Načítanie embeddingov a metadát z cache súboru."""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache súbor {cache_path} neexistuje.")
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata = data["metadata"].tolist()
        logger.info(f"[EmbeddingEngine] Načítaná cache z {cache_path}: {embeddings.shape[0]} chunkov")
        return embeddings, metadata
