"""FinSight RAG (Retrieval-Augmented Generation) package.

Provides modular text chunking, page attribution, local BM25/vector retrieval,
and report-grounded question answering with Gemini.
"""

from rag.chunker import DocumentChunker
from rag.pipeline import RAGPipeline
from rag.qa import QAResult, ReportQAEngine
from rag.retriever import (
    BaseRetriever,
    BM25Retriever,
    VectorRetriever,
    jaccard_similarity,
    tokenize,
)
from rag.schema import DocumentChunk, RetrievalResult

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DocumentChunk",
    "DocumentChunker",
    "QAResult",
    "RAGPipeline",
    "ReportQAEngine",
    "RetrievalResult",
    "VectorRetriever",
    "jaccard_similarity",
    "tokenize",
]
