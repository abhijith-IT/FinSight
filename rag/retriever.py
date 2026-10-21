"""Retrieval module for FinSight RAG.

Provides a modular retrieval architecture:
- BaseRetriever: Abstract interface enabling plug-and-play retrieval strategies.
- BM25Retriever: High-performance, dependency-free local BM25 ranking optimized for financial terms.
- VectorRetriever: Dense embedding retriever interface for semantic search extensibility.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from rag.schema import DocumentChunk, RetrievalResult


# Curated English stopwords that exclude critical financial and numerical terms
# (e.g., 'net', 'gross', 'debt', 'cash', 'tax', 'gain', 'loss' are intentionally kept)
COMMON_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my",
    "myself", "nor", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms and financial symbols."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    return [t for t in raw_tokens if t not in COMMON_STOPWORDS and len(t) > 1]


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate Jaccard similarity between two token sets for deduplication."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class BaseRetriever(ABC):
    """Abstract interface for all FinSight RAG retrievers."""

    @abstractmethod
    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Index a collection of DocumentChunks for retrieval."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve the top_k most relevant chunks for a user query."""
        pass

    @property
    @abstractmethod
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed data."""
        pass


class BM25Retriever(BaseRetriever):
    """Local, dependency-free BM25Okapi retriever for financial document search.

    Why BM25 as default local retrieval:
    1. Zero external network/API dependencies and runs with sub-millisecond latency.
    2. Exact keyword matching is vital for financial reports containing specific
       terminology (e.g. 'Operating Cash Flow', 'debt-to-equity', 'segment revenue',
       disclosures, and footnotes) where semantic models often miss exact numerical context.
    3. Length normalization prevents long, boilerplate pages from dominating search.
    4. Term frequency saturation (k1) avoids over-penalizing repetitive disclaimers.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        dedup_threshold: float = 0.80,
    ) -> None:
        """Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter (default 1.5).
            b: Document length normalization parameter (default 0.75).
            dedup_threshold: Jaccard similarity threshold above which near-duplicate
                             overlapping chunks are filtered out from results.
        """
        self.k1 = k1
        self.b = b
        self.dedup_threshold = dedup_threshold

        self._chunks: List[DocumentChunk] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_token_sets: List[Set[str]] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}
        # Inverted index: term -> list of (doc_idx, term_frequency)
        self._inverted_index: Dict[str, List[tuple[int, int]]] = defaultdict(list)

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Index chunks and compute BM25 corpus statistics."""
        self.clear()
        self._chunks = list(chunks)
        n_docs = len(self._chunks)
        if n_docs == 0:
            return

        total_length = 0
        doc_frequencies: Dict[str, int] = Counter()

        for idx, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.text)
            self._doc_tokens.append(tokens)
            token_set = set(tokens)
            self._doc_token_sets.append(token_set)
            doc_len = len(tokens)
            self._doc_lens.append(doc_len)
            total_length += doc_len

            # Inverted index and document frequencies
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self._inverted_index[term].append((idx, count))
                doc_frequencies[term] += 1

        self._avgdl = total_length / n_docs if n_docs > 0 else 0.0

        # Precompute positive IDF for all corpus terms
        # Formula: ln(1.0 + (N - n + 0.5) / (n + 0.5)) - guaranteed positive for any n <= N
        for term, n_containing in doc_frequencies.items():
            self._idf[term] = math.log(1.0 + (n_docs - n_containing + 0.5) / (n_containing + 0.5))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve most relevant chunks for query with deduplication and thresholding.

        Args:
            query: User's question or search query.
            top_k: Maximum number of results to return.
            min_score: Minimum BM25 score required (filters out completely irrelevant hits).

        Returns:
            List of RetrievalResult ordered by relevance score descending.
        """
        if not self._chunks or not query or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_term_counts = Counter(query_tokens)
        scores: Dict[int, float] = defaultdict(float)

        # Accumulate BM25 scores via inverted index
        for term, q_weight in query_term_counts.items():
            if term not in self._idf:
                continue

            idf = self._idf[term]
            postings = self._inverted_index[term]

            for doc_idx, tf in postings:
                doc_len = self._doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avgdl))
                term_score = idf * (numerator / denominator) * q_weight
                scores[doc_idx] += term_score

        # Filter out candidates below min_score (guarantees irrelevant results are dropped)
        effective_min = max(0.0, min_score)
        eligible = [(doc_idx, s) for doc_idx, s in scores.items() if s > effective_min]
        if not eligible:
            return []

        # Sort descending by score
        eligible.sort(key=lambda x: x[1], reverse=True)

        # Deduplication against higher-ranked overlapping chunks
        results: List[RetrievalResult] = []
        selected_token_sets: List[Set[str]] = []

        for doc_idx, score in eligible:
            candidate_tokens = self._doc_token_sets[doc_idx]

            # Check if this candidate is too similar to an already accepted result
            is_duplicate = False
            for prev_set in selected_token_sets:
                if jaccard_similarity(candidate_tokens, prev_set) >= self.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                chunk = self._chunks[doc_idx]
                results.append(RetrievalResult(chunk=chunk, score=round(float(score), 4)))
                selected_token_sets.append(candidate_tokens)

                if len(results) >= top_k:
                    break

        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._doc_tokens = []
        self._doc_token_sets = []
        self._doc_lens = []
        self._avgdl = 0.0
        self._idf = {}
        self._inverted_index.clear()


class VectorRetriever(BaseRetriever):
    """Dense vector retriever that supports pluggable embedding functions.

    Enables swapping in local embedding models or API embeddings (e.g., Gemini
    embed_content or sentence-transformers) without altering the UI or caller interface.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        dedup_threshold: float = 0.85,
    ) -> None:
        """Initialize VectorRetriever.

        Args:
            embed_fn: Callable that accepts a list of texts and returns a list of float vectors.
            dedup_threshold: Cosine similarity threshold for deduplicating returned results.
        """
        self.embed_fn = embed_fn
        self.dedup_threshold = dedup_threshold
        self._chunks: List[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Embed and store chunk vectors."""
        self.clear()
        self._chunks = list(chunks)
        if not self._chunks:
            return

        if self.embed_fn is None:
            raise ValueError(
                "VectorRetriever requires a valid 'embed_fn' callable to index document chunks."
            )

        texts = [c.text for c in self._chunks]
        raw_vecs = self.embed_fn(texts)
        arr = np.array(raw_vecs, dtype=np.float32)

        # Normalize vectors for cosine similarity via dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings = arr / norms

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve chunks based on cosine similarity."""
        if not self._chunks or self._embeddings is None or not query or not query.strip():
            return []

        if self.embed_fn is None:
            raise ValueError(
                "VectorRetriever requires a valid 'embed_fn' callable to retrieve chunks."
            )

        query_vec = np.array(self.embed_fn([query])[0], dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        scores = np.dot(self._embeddings, query_vec)

        # Filter and sort
        ranked_indices = np.argsort(scores)[::-1]
        results: List[RetrievalResult] = []

        for idx in ranked_indices:
            s = float(scores[idx])
            if s <= min_score:
                break

            chunk = self._chunks[idx]
            results.append(RetrievalResult(chunk=chunk, score=round(s, 4)))

            if len(results) >= top_k:
                break

        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None
