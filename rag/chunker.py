"""Document chunking module for FinSight RAG.

Splits parsed PDF pages into meaningful, overlapping chunks while preserving
strict 1-indexed page number attribution and source metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from ingestion.pdf_parser import ParsedPage, ParsedPDF
from rag.schema import DocumentChunk


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


class DocumentChunker:
    """Configurable text chunker with overlap and page attribution."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 20,
        separators: Optional[List[str]] = None,
    ) -> None:
        """Initialize chunker parameters.

        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
            min_chunk_size: Minimum character length for a chunk to be retained.
            separators: Hierarchical list of separators to split text on.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or list(DEFAULT_SEPARATORS)

    def chunk_page(
        self,
        page: ParsedPage,
        doc_id: str = "doc",
    ) -> List[DocumentChunk]:
        """Split a single ParsedPage into overlapping DocumentChunk objects.

        Preserves the original 1-indexed page_number from ParsedPage.

        Args:
            page: ParsedPage instance from ingestion.
            doc_id: Identifier used in generated chunk_id.

        Returns:
            List of DocumentChunk objects, or empty list if page has no text.
        """
        if not page or not page.has_text or not page.text or not page.text.strip():
            return []

        raw_chunks = self._split_text(page.text.strip())
        document_chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            # Clean and normalize whitespace within chunk
            cleaned_text = re.sub(r"[ \t]+", " ", chunk_text).strip()
            if not cleaned_text:
                continue

            chunk_id = f"{doc_id}_p{page.page_number}_c{idx}"
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    page_number=page.page_number,
                    text=cleaned_text,
                    char_count=len(cleaned_text),
                    token_count=len(cleaned_text.split()),
                    chunk_index=idx,
                    metadata={"page": page.page_number, "doc_id": doc_id},
                )
            )

        return document_chunks

    def chunk_pdf(
        self,
        parsed_pdf: Optional[ParsedPDF],
        doc_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk an entire ParsedPDF across all pages.

        Handles empty documents and blank pages safely.

        Args:
            parsed_pdf: Complete parsed PDF document.
            doc_id: Optional document ID; defaults to parsed_pdf.file_name stem or 'doc'.

        Returns:
            List of all DocumentChunk objects preserving respective page numbers.
        """
        if not parsed_pdf or not parsed_pdf.pages:
            return []

        effective_doc_id = "doc"
        if doc_id:
            effective_doc_id = doc_id
        elif parsed_pdf.file_name:
            effective_doc_id = Path(parsed_pdf.file_name).stem

        all_chunks: List[DocumentChunk] = []
        for page in parsed_pdf.pages:
            page_chunks = self.chunk_page(page, doc_id=effective_doc_id)
            all_chunks.extend(page_chunks)

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping pieces respecting natural boundaries.

        Args:
            text: Non-empty string to split.

        Returns:
            List of chunk strings.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                # Seek a natural break point looking backward from end
                min_bound = start + max(1, self.chunk_overlap)
                for sep in self.separators:
                    pos = text.rfind(sep, min_bound, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunk_str = text[start:end].strip()
            if chunk_str:
                if len(chunk_str) >= self.min_chunk_size or not chunks:
                    chunks.append(chunk_str)

            if end >= text_len:
                break

            # Step back by chunk_overlap to create overlap in the next chunk
            next_start = max(start + 1, end - self.chunk_overlap)

            # Snap next_start forward to next space/newline to prevent breaking mid-word
            space_pos = text.find(" ", next_start, end)
            if space_pos != -1:
                next_start = space_pos + 1

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks
