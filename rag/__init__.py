"""FinSight RAG (Retrieval-Augmented Generation) package.

Provides modular text chunking, page attribution, dense vector embeddings,
hybrid retrieval (BM25 + Semantic), and report-grounded question answering with Gemini.
"""

from rag.chunker import DocumentChunker
from rag.pipeline import RAGPipeline, create_rag_pipeline
from rag.qa import QAResult, ReportQAEngine
from rag.retriever import (
    BaseRetriever,
    BM25Retriever,
    EmbeddingCache,
    GeminiEmbeddingGenerator,
    GLOBAL_EMBEDDING_CACHE,
    HybridRetriever,
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
    "EmbeddingCache",
    "GLOBAL_EMBEDDING_CACHE",
    "GeminiEmbeddingGenerator",
    "HybridRetriever",
    "QAResult",
    "RAGPipeline",
    "ReportQAEngine",
    "RetrievalResult",
    "VectorRetriever",
    "create_rag_pipeline",
    "jaccard_similarity",
    "tokenize",
]
