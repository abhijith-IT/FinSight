"""Retrieval module for FinSight RAG.

Provides a modular retrieval architecture:
- BaseRetriever: Abstract interface enabling plug-and-play retrieval strategies.
- BM25Retriever: High-performance, dependency-free local BM25 ranking optimized for financial terms.
- VectorRetriever: Dense embedding retriever with cosine similarity and deduplication.
- HybridRetriever: Combined keyword and semantic retrieval via Reciprocal Rank Fusion (RRF).
- EmbeddingCache: SHA-256 in-memory cache to prevent redundant embedding computation.
- GeminiEmbeddingGenerator: Batch-oriented embedding generator using google-genai SDK.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from google import genai

import config
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


class EmbeddingCache:
    """Thread-safe in-memory cache mapping text SHA-256 hash to embedding vectors."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[float]] = {}

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute SHA-256 digest of normalized text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding for text, or None if not cached."""
        return self._cache.get(self._hash_text(text))

    def set(self, text: str, vector: Sequence[float]) -> None:
        """Store embedding vector for text in cache."""
        self._cache[self._hash_text(text)] = list(vector)

    def contains(self, text: str) -> bool:
        """Check whether embedding exists for text."""
        return self._hash_text(text) in self._cache

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)


# Global singleton cache for embedding reuse across queries and pipeline runs
GLOBAL_EMBEDDING_CACHE = EmbeddingCache()


class GeminiEmbeddingGenerator:
    """Batch-oriented embedding generator with caching and exponential backoff retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
        cache: Optional[EmbeddingCache] = None,
        batch_size: int = 50,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
    ) -> None:
        """Initialize the Gemini embedding generator.

        Args:
            api_key: Gemini API key. Defaults to config.GEMINI_API_KEY.
            model_name: Embedding model name. Defaults to config.GEMINI_EMBEDDING_MODEL or 'text-embedding-004'.
            client: Optional pre-configured genai.Client instance.
            cache: Optional EmbeddingCache; defaults to GLOBAL_EMBEDDING_CACHE.
            batch_size: Maximum texts per batch request.
            max_retries: Maximum retry attempts for transient errors.
            base_delay_seconds: Base backoff delay for retries.
        """
        self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
        self.model_name = (
            model_name
            or getattr(config, "GEMINI_EMBEDDING_MODEL", None)
            or "text-embedding-004"
        )
        self._client = client
        self.cache = cache if cache is not None else GLOBAL_EMBEDDING_CACHE
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds

    @property
    def client(self) -> genai.Client:
        """Get or initialize the google-genai Client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("Gemini API key is required to generate embeddings.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embedding vectors for a sequence of texts using caching and batching.

        Checks cache first. Only un-cached texts trigger Gemini API requests.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        # 1. Check cache for each text
        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        # 2. Batch and request missing embeddings
        if missing_texts:
            for b_start in range(0, len(missing_texts), self.batch_size):
                b_end = min(b_start + self.batch_size, len(missing_texts))
                batch_slice = missing_texts[b_start:b_end]
                batch_indices = missing_indices[b_start:b_end]

                batch_vectors = self._embed_batch_with_retry(batch_slice)
                for orig_idx, text, vec in zip(batch_indices, batch_slice, batch_vectors):
                    self.cache.set(text, vec)
                    results[orig_idx] = vec

        # Fill any missing with zero vectors as fallback (should not normally occur)
        dim = len(results[0]) if results and results[0] else 768
        return [v if v is not None else [0.0] * dim for v in results]

    def _embed_batch_with_retry(self, batch_texts: List[str]) -> List[List[float]]:
        """Execute batch embedding request with bounded exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch_texts if len(batch_texts) > 1 else batch_texts[0],
                )
                embeddings = getattr(response, "embeddings", None) or []
                return [list(emb.values) for emb in embeddings]
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(
                    code in err_str for code in ["503", "429", "unavailable", "resource_exhausted", "timeout"]
                )
                if attempt < self.max_retries and is_transient:
                    delay = self.base_delay_seconds * (2 ** attempt) + random.uniform(0.1, 0.4)
                    time.sleep(delay)
                    continue
                raise last_error
        raise last_error or RuntimeError("Failed to embed batch")

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        """Enable callable interface compatible with VectorRetriever.embed_fn."""
        return self.embed_texts(texts)


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
    """Dense vector retriever supporting pluggable embedding functions.

    Features:
    - Lightweight, in-memory dense vector store using NumPy.
    - Cosine similarity ranking via L2 normalization.
    - Vector-space deduplication to filter redundant overlapping chunks.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        dedup_threshold: float = 0.90,
    ) -> None:
        """Initialize VectorRetriever.

        Args:
            embed_fn: Callable accepting a list of texts and returning a list of float vectors.
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

        raw_q_vec = self.embed_fn([query])[0]
        query_vec = np.array(raw_q_vec, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        scores = np.dot(self._embeddings, query_vec)

        # Filter and sort descending with stable sort
        ranked_indices = np.argsort(-scores, kind="stable")
        results: List[RetrievalResult] = []
        selected_vectors: List[np.ndarray] = []

        for idx in ranked_indices:
            s = float(scores[idx])
            if s <= min_score:
                break

            cand_vec = self._embeddings[idx]

            # Vector-space deduplication: skip candidate if cosine similarity to already accepted is high
            is_dup = False
            for prev_vec in selected_vectors:
                sim = float(np.dot(cand_vec, prev_vec))
                if sim >= self.dedup_threshold:
                    is_dup = True
                    break

            if not is_dup:
                chunk = self._chunks[idx]
                results.append(RetrievalResult(chunk=chunk, score=round(s, 4)))
                selected_vectors.append(cand_vec)

                if len(results) >= top_k:
                    break

        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 keyword ranking and dense vector embeddings.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both retrievers without
    requiring arbitrary score scaling or calibration.
    """

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            bm25_retriever: BM25Retriever instance; defaults to new BM25Retriever.
            vector_retriever: VectorRetriever instance.
            bm25_weight: Weight allocated to BM25 ranks.
            vector_weight: Weight allocated to Vector ranks.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (default 60).
        """
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.vector_retriever = vector_retriever
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self._chunks: List[DocumentChunk] = []

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Index chunks across both retrievers."""
        self._chunks = list(chunks)
        self.bm25_retriever.index(chunks)
        if self.vector_retriever and self.vector_retriever.embed_fn:
            self.vector_retriever.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve top_k chunks using Reciprocal Rank Fusion of BM25 and Vector search."""
        if not self._chunks or not query or not query.strip():
            return []

        # If vector retriever is not available or has no chunks, delegate directly to BM25
        if not self.vector_retriever or self.vector_retriever.chunk_count == 0:
            return self.bm25_retriever.retrieve(query=query, top_k=top_k, min_score=min_score)

        pool_size = max(top_k * 3, 20)
        bm25_hits = self.bm25_retriever.retrieve(query=query, top_k=pool_size, min_score=0.0)

        try:
            vector_hits = self.vector_retriever.retrieve(query=query, top_k=pool_size, min_score=0.0)
        except Exception:
            # Fall back gracefully to BM25 if vector retriever fails during search
            return self.bm25_retriever.retrieve(query=query, top_k=top_k, min_score=min_score)

        if not bm25_hits and not vector_hits:
            return []

        # Reciprocal Rank Fusion (RRF)
        # score = sum(weight / (k + rank))
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit.chunk_id
            rrf_scores[cid] += self.bm25_weight / (self.rrf_k + rank)
            chunk_map[cid] = hit.chunk

        for rank, hit in enumerate(vector_hits, start=1):
            cid = hit.chunk_id
            rrf_scores[cid] += self.vector_weight / (self.rrf_k + rank)
            chunk_map[cid] = hit.chunk

        # Sort descending by fused score
        ranked_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build deduplicated results
        results: List[RetrievalResult] = []
        selected_token_sets: List[Set[str]] = []

        for cid, score in ranked_items:
            chunk = chunk_map[cid]
            tokens = set(tokenize(chunk.text))

            is_dup = False
            for prev_tokens in selected_token_sets:
                if jaccard_similarity(tokens, prev_tokens) >= 0.80:
                    is_dup = True
                    break

            if not is_dup:
                results.append(RetrievalResult(chunk=chunk, score=round(float(score), 5)))
                selected_token_sets.append(tokens)

                if len(results) >= top_k:
                    break

        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks = []
        self.bm25_retriever.clear()
        if self.vector_retriever:
            self.vector_retriever.clear()
