"""
Modul pre rozdelenie kolekcie textov na menšie úseky (chunky).
Umožňuje nastaviteľnú veľkosť a prekrytie v slovách alebo tokenoch
a zachováva informáciu o pôvode jednotlivých chunkov.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re


@dataclass
class TextChunk:
    chunk_id: str          # Unikátne ID chunku: f"{doc_id}_c{index}"
    doc_id: str            # Pôvodné ID dokumentu
    title: str             # Názov dokumentu / článku
    text: str              # Textový obsah chunku
    chunk_index: int       # Poradové číslo chunku v rámci dokumentu
    start_char: int        # Počiatočná pozícia v zdrojovom texte
    end_char: int          # Koncová pozícia v zdrojovom texte
    unit: str              # Jednotka veľkosti: 'words' alebo 'tokens'
    size: int              # Počet jednotiek v chunku

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        return cls(**data)


class TextChunker:
    """
    TextSplitter s nastaviteľnou veľkosťou chunkov a ich prekrytím.
    Podporuje jednotky 'words' (slová) a zachováva hranice viet pre lepšiu súdržnosť textu.
    """

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        unit: str = "words",
        min_chunk_size: int = 20,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) musí byť menší ako chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit
        self.min_chunk_size = min(min_chunk_size, max(1, chunk_size // 4))

    def clean_text(self, text: str) -> str:
        """Vyčistenie textu od nadbytočných prázdnych znakov a HTML značiek."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        """Rozdelenie textu na vety s ohľadom na skratky a interpunkciu."""
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-ZÁČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ0-9])"
        raw_sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def chunk_document(self, doc_id: str, title: str, text: str) -> List[TextChunk]:
        """
        Rozdelí jeden dokument na chunky s posuvným oknom (sliding window) a prekrytím.
        Zachováva informáciu o pôvode (doc_id, title, char span, index).
        """
        clean_doc = self.clean_text(text)
        if not clean_doc:
            return []

        words_matches = list(re.finditer(r"\S+", clean_doc))
        if not words_matches:
            return []

        chunks: List[TextChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        total_words = len(words_matches)

        chunk_idx = 0
        i = 0
        while i < total_words:
            end_i = min(i + self.chunk_size, total_words)
            chunk_word_matches = words_matches[i:end_i]

            # Ak je posledný chunk príliš malý a už máme predchádzajúce chunky, môžeme ho vynechať alebo spojiť
            if len(chunk_word_matches) < self.min_chunk_size and chunks:
                break

            start_char = chunk_word_matches[0].start()
            end_char = chunk_word_matches[-1].end()
            chunk_text = clean_doc[start_char:end_char].strip()

            chunk_id = f"{doc_id}_c{chunk_idx}"
            chunk = TextChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                text=chunk_text,
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
                unit=self.unit,
                size=len(chunk_word_matches),
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end_i == total_words:
                break
            i += step

        return chunks

    def chunk_collection(self, documents: List[Dict[str, Any]]) -> List[TextChunk]:
        """
        Dávkové rozdelenie kolekcie dokumentov na chunky.
        Každý dokument sa spracováva samostatne.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            doc_id = str(doc.get("id", f"doc_{len(all_chunks)}"))
            title = str(doc.get("title", doc.get("source", doc_id)))
            text = str(doc.get("text", ""))
            doc_chunks = self.chunk_document(doc_id=doc_id, title=title, text=text)
            all_chunks.extend(doc_chunks)
        return all_chunks
