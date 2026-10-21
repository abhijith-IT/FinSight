"""Data models for FinSight RAG (Retrieval-Augmented Generation) pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """Structured representation of a text chunk extracted from a document page."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Unique identifier for the chunk (e.g. doc_p1_c0)")
    page_number: int = Field(..., ge=1, description="1-indexed source PDF page number")
    text: str = Field(..., min_length=1, description="Text content of this chunk")
    char_count: int = Field(..., ge=0, description="Character count of the chunk text")
    token_count: int = Field(default=0, ge=0, description="Estimated token count (or word count)")
    chunk_index: int = Field(default=0, ge=0, description="0-indexed position of chunk within page")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the chunk (file name, section, etc.)",
    )


class RetrievalResult(BaseModel):
    """Structured result returned by a retriever for a query."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk = Field(..., description="The matched document chunk")
    score: float = Field(..., description="Relevance score (higher indicates more relevant)")

    @property
    def page_number(self) -> int:
        """1-indexed source page number."""
        return self.chunk.page_number

    @property
    def text(self) -> str:
        """Source text content."""
        return self.chunk.text

    @property
    def chunk_id(self) -> str:
        """Chunk unique ID."""
        return self.chunk.chunk_id
