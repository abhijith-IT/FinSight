"""End-to-end RAG pipeline for FinSight.

Coordinates document chunking and retrieval into a unified, modular interface
that works seamlessly with ParsedPDF inputs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ingestion.pdf_parser import ParsedPDF
from rag.chunker import DocumentChunker
from rag.retriever import BaseRetriever, BM25Retriever
from rag.schema import DocumentChunk, RetrievalResult


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

    def ingest(
        self,
        parsed_pdf: Optional[ParsedPDF],
        doc_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk and index a parsed PDF.

        Handles empty documents safely without raising errors.

        Args:
            parsed_pdf: ParsedPDF document from ingestion.
            doc_id: Optional identifier for chunk attribution.

        Returns:
            List of generated DocumentChunks.
        """
        chunks = self.chunker.chunk_pdf(parsed_pdf, doc_id=doc_id)
        self._chunks = chunks
        self.retriever.index(chunks)
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

    def clear(self) -> None:
        """Reset chunk storage and clear the retriever index."""
        self._chunks = []
        self.retriever.clear()
