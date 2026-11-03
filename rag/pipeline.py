"""End-to-end RAG pipeline for FinSight.

Coordinates document chunking and retrieval into a unified, modular interface
that works seamlessly with ParsedPDF inputs.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

from google import genai

import config
from ingestion.pdf_parser import ParsedPDF
from rag.chunker import DocumentChunker
from rag.retriever import (
    BaseRetriever,
    BM25Retriever,
    GeminiEmbeddingGenerator,
    HybridRetriever,
    VectorRetriever,
)
from rag.schema import DocumentChunk, RetrievalResult

__all__ = ["RAGPipeline", "create_rag_pipeline"]


class RAGPipeline:
    """High-level pipeline coordinating document chunking and retrieval."""

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        retriever: Optional[BaseRetriever] = None,
    ) -> None:
        """Initialize RAG pipeline with modular components.

        Args:
            chunker: DocumentChunker instance; defaults to standard DocumentChunker.
            retriever: BaseRetriever implementation; defaults to BM25Retriever.
        """
        self.chunker = chunker or DocumentChunker()
        self.retriever = retriever or BM25Retriever()
        self._chunks: List[DocumentChunk] = []
        self._indexed_doc_hash: Optional[str] = None

    @staticmethod
    def _compute_doc_hash(parsed_pdf: ParsedPDF, doc_id: Optional[str] = None) -> str:
        """Compute deterministic digest of parsed document contents to avoid repeated indexing."""
        ident = f"{doc_id or ''}_{parsed_pdf.file_name or ''}_{parsed_pdf.total_characters}_{parsed_pdf.total_pages}"
        p_info = ""
        if parsed_pdf.pages:
            p_info = f"_{parsed_pdf.pages[0].char_count}_{parsed_pdf.pages[-1].char_count}"
        return hashlib.sha256((ident + p_info).encode("utf-8")).hexdigest()

    def ingest(
        self,
        parsed_pdf: Optional[ParsedPDF],
        doc_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk and index a parsed PDF.

        Reuses already indexed chunks if the identical document is passed,
        preventing repeated chunking and embedding operations.

        Args:
            parsed_pdf: ParsedPDF document from ingestion.
            doc_id: Optional identifier for chunk attribution.

        Returns:
            List of generated DocumentChunks.
        """
        if not parsed_pdf or not parsed_pdf.pages:
            self.clear()
            return []

        doc_hash = self._compute_doc_hash(parsed_pdf, doc_id)
        if self._indexed_doc_hash == doc_hash and self.chunk_count > 0:
            return list(self._chunks)

        chunks = self.chunker.chunk_pdf(parsed_pdf, doc_id=doc_id)
        self._chunks = chunks
        self.retriever.index(chunks)
        self._indexed_doc_hash = doc_hash
        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Search the indexed document chunks for the most relevant passages.

        Args:
            query: User's question or search term.
            top_k: Maximum number of relevant chunks to retrieve.
            min_score: Minimum relevance score required.

        Returns:
            List of RetrievalResult with source text and 1-indexed page numbers.
        """
        return self.retriever.retrieve(query=query, top_k=top_k, min_score=min_score)

    @property
    def chunks(self) -> List[DocumentChunk]:
        """Return a copy of the currently indexed document chunks."""
        return list(self._chunks)

    @property
    def chunk_count(self) -> int:
        """Total number of indexed chunks."""
        return len(self._chunks)

    @property
    def indexed_pages(self) -> List[int]:
        """Sorted list of unique 1-indexed page numbers present in the index."""
        return sorted({c.page_number for c in self._chunks})

    @property
    def is_semantic_unavailable(self) -> bool:
        """Check if dense vector retrieval is unavailable due to embedding failure."""
        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.semantic_unavailable
        return False

    @property
    def embedding_error(self) -> Optional[str]:
        """Return the embedding error message if semantic retrieval is unavailable."""
        if isinstance(self.retriever, HybridRetriever):
            return self.retriever.embedding_error
        return None

    def clear(self) -> None:
        """Reset chunk storage and clear the retriever index."""
        self._chunks = []
        self._indexed_doc_hash = None
        self.retriever.clear()


def create_rag_pipeline(
    api_key: Optional[str] = None,
    client: Optional[genai.Client] = None,
    retriever_type: str = "hybrid",
    embedding_model: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> RAGPipeline:
    """Factory creating a fully configured RAGPipeline.

    Args:
        api_key: Gemini API key for embedding generation.
        client: Pre-configured genai.Client instance.
        retriever_type: Retrieval strategy: 'hybrid', 'vector', or 'bm25'.
        embedding_model: Embedding model name; defaults to config.GEMINI_EMBEDDING_MODEL.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        Configured RAGPipeline instance.
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    has_api_access = bool(api_key or client or getattr(config, "GEMINI_API_KEY", None))

    if retriever_type in ("hybrid", "vector") and has_api_access:
        embed_gen = GeminiEmbeddingGenerator(
            api_key=api_key,
            client=client,
            model_name=embedding_model,
        )
        vec_retriever = VectorRetriever(embed_fn=embed_gen)

        if retriever_type == "vector":
            retriever: BaseRetriever = vec_retriever
        else:
            bm25 = BM25Retriever()
            retriever = HybridRetriever(bm25_retriever=bm25, vector_retriever=vec_retriever)
    else:
        # Default or fallback to local BM25 retriever
        retriever = BM25Retriever()

    return RAGPipeline(chunker=chunker, retriever=retriever)
