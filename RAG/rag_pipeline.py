"""
Modul pre RAG pipeline a integráciu generatívneho jazykového modelu (< 10B parametrov).
Podporuje modely cez HuggingFace Transformers (napr. Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct)
aj cez lokálnu Ollama službu.
"""

from typing import List, Dict, Any, Optional
import os
import logging
import json
import importlib
import urllib.request
import urllib.error
from chunker import TextChunk
from retriever import RetrievalResult


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """Si presný a pravdovravný asistent pre zodpovedanie otázok v slovenskom jazyku.
Tvojou úlohou je odpovedať na položenú otázku VÝHRADNE na základe poskytnutého kontextu.

Dôležité inštrukcie:
1. Odpovedaj stručne, vecne a presne v slovenskom jazyku.
2. Ak poskytnutý kontext neobsahuje informácie potrebné na zodpovedanie otázky, napíš:
   "Na základe poskytnutého kontextu nie je možné na túto otázku odpovedať."
3. Za žiadnych okolností si nevymýšľaj fakty ani nepoužívaj neoverené externé znalosti, ktoré v kontexte nie sú.
"""


class RAGPipeline:
    """
    RAG Pipeline spájajúca systémový prompt, vyhľadané chunky a generatívny LLM model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        backend: str = "transformers",  # "transformers" alebo "ollama"
        ollama_url: str = "http://localhost:11434",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: Optional[str] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.backend = backend.lower()
        self.ollama_url = ollama_url
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch = None

        if self.backend == "transformers":
            try:
                self.torch = importlib.import_module("torch")
            except ImportError as e:
                raise ImportError(
                    "Backend 'transformers' vyžaduje balík torch. "
                    "Nainštalujte ho podľa pokynov na https://pytorch.org/get-started/locally/"
                ) from e

        if device is None:
            self.device = "cuda" if self.torch and self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.hf_model = None
        self.hf_tokenizer = None

        if self.backend == "transformers":
            self._init_transformers()
        elif self.backend == "ollama":
            self._check_ollama()
        else:
            raise ValueError(f"Neznámy backend: {backend}. Použite 'transformers' alebo 'ollama'.")

    def _init_transformers(self):
        """Inicializácia lokálneho Hugging Face modelu."""
        if self.torch is None:
            raise ImportError("Backend 'transformers' vyžaduje balík torch.")

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as e:
            raise ImportError(
                "Backend 'transformers' vyžaduje balík transformers. "
                "Nainštalujte ho príkazom: pip install transformers"
            ) from e

        logger.info(f"[RAGPipeline] Načítavam generatívny model {self.model_name} (Transformers na {self.device})...")
        self.hf_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        
        # Nastavenie torch_dtype
        dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
        self.hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device != "cuda":
            self.hf_model = self.hf_model.to(self.device)
        self.hf_model.eval()

        # Zistenie počtu parametrov
        total_params = sum(p.numel() for p in self.hf_model.parameters())
        self.param_count_str = f"{total_params / 1e9:.2f}B parametrov"
        logger.info(f"[RAGPipeline] Model načítaný: {self.model_name} ({self.param_count_str})")

    def _check_ollama(self):
        """Overenie dostupnosti Ollama servera."""
        self.param_count_str = "< 10B parametrov (Ollama)"
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                logger.info(f"[RAGPipeline] Ollama pripojená úspešne. Dostupné modely: {models}")
        except Exception as e:
            logger.warning(f"[RAGPipeline] UPOZORNENIE: Ollama na {self.ollama_url} neodpovedá ({e}).")
            logger.warning("Tip: Spustite `ollama serve` v termináli, alebo použite backend 'transformers'.")

    def format_rag_context(self, retrieved_results: List[RetrievalResult]) -> str:
        """Sformátovanie textov nájdených chunkov s označením pôvodu a metadát."""
        if not retrieved_results:
            return "Žiadny kontext nie je k dispozícii."

        context_blocks = []
        for i, res in enumerate(retrieved_results, 1):
            ch = res.chunk
            header = f"--- [Úryvok #{i} | Zdroj: {ch.title} (ID: {ch.doc_id}), Časť #{ch.chunk_index}, Podobnosť: {res.score:.3f}] ---"
            context_blocks.append(f"{header}\n{ch.text.strip()}")

        return "\n\n".join(context_blocks)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Zavolanie jazykového modelu (Transformers alebo Ollama)."""
        sys_p = system_prompt or self.system_prompt

        if self.backend == "transformers":
            if self.hf_tokenizer is None:
                raise RuntimeError("Transformers backend nie je správne inicializovaný: tokenizer je None.")
            if self.torch is None:
                raise RuntimeError("Transformers backend nie je správne inicializovaný: torch je None.")

            torch_module = self.torch
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt}
            ]
            if hasattr(self.hf_tokenizer, "apply_chat_template"):
                input_text = self.hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = f"{sys_p}\n\nUžívateľ: {prompt}\nAsistent:"

            if self.hf_model is None:
                raise RuntimeError("Transformers backend nie je správne inicializovaný: model je None.")

            inputs = self.hf_tokenizer(input_text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            with torch_module.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    top_p=0.9 if self.temperature > 0 else None,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                )

            gen_tokens = outputs[0][input_len:]
            response = self.hf_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            return response

        elif self.backend == "ollama":
            # Ollama /api/chat endpoint
            url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens
                }
            }
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data_bytes, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data.get("message", {}).get("content", "").strip()

        return ""

    def answer_with_rag(self, question: str, retrieved_results: List[RetrievalResult]) -> Dict[str, Any]:
        """Kompletný RAG prechod: otázka + vyhľadané chunky -> LLM odpoveď."""
        context_str = self.format_rag_context(retrieved_results)
        user_prompt = f"Kontext:\n{context_str}\n\nOtázka: {question}\n\nOdpovedz stručne na základe kontextu:"
        answer = self.generate_response(prompt=user_prompt, system_prompt=self.system_prompt)
        return {
            "question": question,
            "answer": answer,
            "context": context_str,
            "chunks_used": [r.chunk.chunk_id for r in retrieved_results],
        }

    def answer_pure_llm(self, question: str) -> Dict[str, Any]:
        """Samostatný jazykový model bez prístupu ku kolekcii (Zero-shot)."""
        sys_p = "Si užitočný asistent. Odpovedaj na otázky stručne a vecne v slovenskom jazyku."
        user_prompt = f"Otázka: {question}\n\nOdpovedz na túto otázku priamo a stručne v slovenskom jazyku:"
        answer = self.generate_response(prompt=user_prompt, system_prompt=sys_p)
        return {
            "question": question,
            "answer": answer,
        }
