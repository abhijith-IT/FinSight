"""PDF text extraction module for FinSight.

Extracts text from financial reports and documents using pdfplumber,
returning structured page-level data with robust error handling.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import pdfplumber
from pydantic import BaseModel, ConfigDict, Field


class PDFParserError(Exception):
    """Base exception for all PDF parsing errors."""
    pass


class InvalidPDFError(PDFParserError):
    """Raised when the input is corrupted, not a valid PDF, or cannot be parsed."""
    pass


class EmptyPDFError(PDFParserError):
    """Raised when the input PDF has 0 bytes or contains no pages."""
    pass


class ParsedPage(BaseModel):
    """Structured representation of an extracted PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(default="", description="Extracted text content of the page")
    has_text: bool = Field(default=False, description="Flag indicating if extractable text is present")
    char_count: int = Field(default=0, description="Character count of extracted text")


class ParsedPDF(BaseModel):
    """Structured representation of the complete parsed PDF document."""

    model_config = ConfigDict(frozen=True)

    file_name: Optional[str] = Field(default=None, description="Name or identifier of the source file")
    total_pages: int = Field(..., description="Total number of pages in the PDF")
    pages: List[ParsedPage] = Field(default_factory=list, description="Extracted page-by-page data")
    total_characters: int = Field(default=0, description="Total extracted characters across all pages")

    @property
    def full_text(self) -> str:
        """Returns concatenated text of all non-empty pages separated by two newlines."""
        return "\n\n".join(page.text for page in self.pages if page.has_text)


class PDFParser:
    """Modular parser for extracting page-level text from PDF documents."""

    def __init__(self, clean_text: bool = True) -> None:
        """Initialize parser options.

        Args:
            clean_text: Whether to strip leading and trailing whitespace from each page.
        """
        self.clean_text = clean_text

    def parse(self, source: Union[str, Path, BinaryIO, bytes]) -> ParsedPDF:
        """Parse text from a PDF file path, byte stream, or raw bytes.

        Args:
            source: Path to PDF file, open binary file-like object, or raw bytes.

        Returns:
            ParsedPDF: Structured page-by-page extracted data and metadata.

        Raises:
            FileNotFoundError: If a file path is specified but does not exist.
            EmptyPDFError: If the source contains 0 bytes or 0 pages.
            InvalidPDFError: If the PDF is corrupted or invalid.
        """
        file_name: Optional[str] = None
        stream_or_path: Union[str, BinaryIO]

        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            if not path_obj.exists():
                raise FileNotFoundError(f"PDF file does not exist: {path_obj}")
            if not path_obj.is_file():
                raise InvalidPDFError(f"Specified path is not a file: {path_obj}")
            if path_obj.stat().st_size == 0:
                raise EmptyPDFError(f"PDF file is empty (0 bytes): {path_obj}")
            file_name = path_obj.name
            stream_or_path = str(path_obj)

        elif isinstance(source, bytes):
            if len(source) == 0:
                raise EmptyPDFError("Provided PDF byte buffer is empty (0 bytes).")
            stream_or_path = io.BytesIO(source)

        elif hasattr(source, "read"):
            file_name = getattr(source, "name", None)
            if file_name is not None:
                file_name = Path(str(file_name)).name

            # Check for empty stream
            current_pos = source.tell() if hasattr(source, "tell") else 0
            if hasattr(source, "seek"):
                source.seek(0, os.SEEK_END)
                size = source.tell()
                source.seek(current_pos)
                if size == 0:
                    raise EmptyPDFError("Provided PDF stream is empty (0 bytes).")

            stream_or_path = source

        else:
            raise InvalidPDFError(f"Unsupported source type: {type(source).__name__}")

        try:
            with pdfplumber.open(stream_or_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    raise EmptyPDFError("The PDF document contains 0 pages.")

                parsed_pages: List[ParsedPage] = []
                for idx, page in enumerate(pdf.pages, start=1):
                    try:
                        raw_text = page.extract_text()
                    except Exception as extract_err:
                        # Gracefully handle unexpected per-page extraction failures
                        raw_text = None

                    if raw_text is None:
                        text = ""
                    elif self.clean_text:
                        text = raw_text.strip()
                    else:
                        text = raw_text

                    has_text = bool(text and not text.isspace())
                    parsed_pages.append(
                        ParsedPage(
                            page_number=idx,
                            text=text,
                            has_text=has_text,
                            char_count=len(text),
                        )
                    )

                total_chars = sum(p.char_count for p in parsed_pages)

                return ParsedPDF(
                    file_name=file_name,
                    total_pages=total_pages,
                    pages=parsed_pages,
                    total_characters=total_chars,
                )

        except (EmptyPDFError, FileNotFoundError):
            raise
        except Exception as e:
            raise InvalidPDFError(f"Failed to parse PDF document: {e}") from e


def extract_text_from_pdf(
    source: Union[str, Path, BinaryIO, bytes],
    clean_text: bool = True,
) -> ParsedPDF:
    """Convenience function to extract text from a PDF source.

    Args:
        source: File path, Path object, binary stream, or bytes.
        clean_text: Whether to strip leading and trailing whitespace.

    Returns:
        ParsedPDF with structured page-level text data.
    """
    return PDFParser(clean_text=clean_text).parse(source)
