"""
Section-aware chunking engine for MedDriftGuard.

Splits clean medical text sections into ~250-350 token chunks with 15% boundary overlap.
Assigns deterministic IDs, preserves topic and section context, and deduplicates.
"""

import hashlib
import re
from typing import Dict, List, Optional, Set
from src.common.schemas import RetrievedChunk
from kb.safety_filter import contains_drug_dosage, strip_drug_dosages


class TextChunker:
    """
    Handles section-aware text chunking with deterministic ID generation
    and deduplication.
    """

    def __init__(
        self,
        target_tokens: int = 300,
        min_tokens: int = 40,
        max_tokens: int = 380,
        overlap_pct: float = 0.15,
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = int(target_tokens * overlap_pct)
        self._seen_hashes: Set[str] = set()

    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count based on whitespace word tokenization."""
        return len(text.split())

    @staticmethod
    def slugify(text: str) -> str:
        """Create a clean lowercase slug from topic or section name."""
        clean = re.sub(r"[^\w\s-]", "", text.lower())
        return re.sub(r"[-\s]+", "_", clean).strip("_")

    def chunk_section(
        self,
        source: str,
        url: str,
        date_accessed: str,
        topic: str,
        section_name: str,
        section_text: str,
        license_str: str,
        language: str = "en",
        start_index: int = 1,
    ) -> List[RetrievedChunk]:
        """
        Chunks a single section's text into overlapping chunks, applying safety
        dosage filters and ensuring deterministic chunk_ids.
        """
        # Step 1: Strip any prohibited drug dosages
        cleaned_text, _ = strip_drug_dosages(section_text)

        words = cleaned_text.split()
        if not words:
            return []

        chunks: List[RetrievedChunk] = []
        topic_slug = self.slugify(topic)
        source_slug = self.slugify(source)[:3]

        start = 0
        current_index = start_index

        while start < len(words):
            end = min(start + self.target_tokens, len(words))

            # If the remaining words are too few to form a separate chunk, merge them
            if len(words) - end < self.min_tokens:
                end = len(words)

            chunk_words = words[start:end]
            chunk_body = " ".join(chunk_words)

            # Prepend topic and section context for retrieval grounding
            formatted_text = f"[{topic.title()} | {section_name.title()}]\n{chunk_body}"

            # Deduplication by text content hash
            content_hash = hashlib.sha256(formatted_text.encode("utf-8")).hexdigest()
            if content_hash in self._seen_hashes:
                start += self.target_tokens - self.overlap_tokens
                continue

            self._seen_hashes.add(content_hash)

            # Deterministic chunk ID: e.g. "mp_fever_0001"
            chunk_id = f"{source_slug}_{topic_slug}_{current_index:04d}"

            # Double check safety regex against RetrievedChunk schema
            if not contains_drug_dosage(formatted_text):
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    source=source,
                    url=url,
                    date_accessed=date_accessed,
                    section=section_name,
                    language=language,
                    topic=topic,
                    license=license_str,
                    text=formatted_text,
                )
                chunks.append(chunk)
                current_index += 1

            if end == len(words):
                break

            # Advance window with 15% overlap
            step = max(self.target_tokens - self.overlap_tokens, 1)
            start += step

        return chunks
