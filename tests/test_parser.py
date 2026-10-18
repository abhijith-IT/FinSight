"""Unit tests for FinSight PDF parser module."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pypdfium2 as pdfium

from ingestion.pdf_parser import (
    EmptyPDFError,
    InvalidPDFError,
    ParsedPage,
    ParsedPDF,
    PDFParser,
    PDFParserError,
    extract_text_from_pdf,
)


class TestPDFParser(unittest.TestCase):
    """Test suite for PDFParser and extract_text_from_pdf."""

    def test_extract_text_valid_mocked_pdf(self) -> None:
        """Verify structured extraction across multiple valid pages."""
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "  Balance Sheet\nTotal Assets: $100M  "

        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Income Statement\nRevenue: $50M"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page_1, mock_page_2]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf

            result = extract_text_from_pdf(b"%PDF-mock", clean_text=True)

            self.assertIsInstance(result, ParsedPDF)
            self.assertEqual(result.total_pages, 2)
            self.assertEqual(len(result.pages), 2)

            # Check Page 1
            p1 = result.pages[0]
            self.assertEqual(p1.page_number, 1)
            self.assertEqual(p1.text, "Balance Sheet\nTotal Assets: $100M")
            self.assertTrue(p1.has_text)
            self.assertEqual(p1.char_count, len(p1.text))

            # Check Page 2
            p2 = result.pages[1]
            self.assertEqual(p2.page_number, 2)
            self.assertEqual(p2.text, "Income Statement\nRevenue: $50M")
            self.assertTrue(p2.has_text)

            # Check full text assembly
            expected_full_text = f"{p1.text}\n\n{p2.text}"
            self.assertEqual(result.full_text, expected_full_text)
            self.assertEqual(result.total_characters, len(p1.text) + len(p2.text))

    def test_page_without_extractable_text(self) -> None:
        """Verify handling of pages that contain no text (e.g. scans, images, blank pages)."""
        mock_page_text = MagicMock()
        mock_page_text.extract_text.return_value = "Financial Highlights"

        # None return from pdfplumber indicates no extractable text
        mock_page_none = MagicMock()
        mock_page_none.extract_text.return_value = None

        # Whitespace-only page
        mock_page_whitespace = MagicMock()
        mock_page_whitespace.extract_text.return_value = "   \n\t  "

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page_text, mock_page_none, mock_page_whitespace]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf

            result = extract_text_from_pdf(b"%PDF-mock")

            self.assertEqual(result.total_pages, 3)

            # Page 1: normal text
            self.assertTrue(result.pages[0].has_text)
            self.assertEqual(result.pages[0].text, "Financial Highlights")

            # Page 2: None -> empty string, has_text=False
            self.assertFalse(result.pages[1].has_text)
            self.assertEqual(result.pages[1].text, "")
            self.assertEqual(result.pages[1].char_count, 0)

            # Page 3: Whitespace -> empty string, has_text=False
            self.assertFalse(result.pages[2].has_text)
            self.assertEqual(result.pages[2].text, "")
            self.assertEqual(result.pages[2].char_count, 0)

            # full_text should only contain page 1
            self.assertEqual(result.full_text, "Financial Highlights")

    def test_page_extraction_resilience_on_exception(self) -> None:
        """Ensure an unexpected exception during page extraction is handled gracefully."""
        mock_page_error = MagicMock()
        mock_page_error.extract_text.side_effect = RuntimeError("Decompression failed")

        mock_page_ok = MagicMock()
        mock_page_ok.extract_text.return_value = "Notes to Statements"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page_error, mock_page_ok]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf

            result = extract_text_from_pdf(b"%PDF-mock")

            self.assertEqual(result.total_pages, 2)
            self.assertFalse(result.pages[0].has_text)
            self.assertEqual(result.pages[0].text, "")
            self.assertTrue(result.pages[1].has_text)
            self.assertEqual(result.pages[1].text, "Notes to Statements")

    def test_clean_text_disabled(self) -> None:
        """Verify raw text extraction preserves leading and trailing spaces when clean_text=False."""
        raw_string = "  Unstripped Text \n "
        mock_page = MagicMock()
        mock_page.extract_text.return_value = raw_string

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf

            parser = PDFParser(clean_text=False)
            result = parser.parse(b"%PDF-mock")

            self.assertEqual(result.pages[0].text, raw_string)
            self.assertEqual(result.pages[0].char_count, len(raw_string))

    def test_empty_inputs_raise_empty_pdf_error(self) -> None:
        """Verify that 0-byte inputs raise EmptyPDFError."""
        # Empty bytes
        with self.assertRaises(EmptyPDFError):
            extract_text_from_pdf(b"")

        # Empty BytesIO stream
        with self.assertRaises(EmptyPDFError):
            extract_text_from_pdf(io.BytesIO(b""))

        # Empty file on disk
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with self.assertRaises(EmptyPDFError):
                extract_text_from_pdf(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_pdf_with_zero_pages_raises_empty_pdf_error(self) -> None:
        """Verify that a PDF document with 0 pages raises EmptyPDFError."""
        mock_pdf = MagicMock()
        mock_pdf.pages = []

        with patch("pdfplumber.open") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_pdf

            with self.assertRaises(EmptyPDFError):
                extract_text_from_pdf(b"%PDF-valid-headers-zero-pages")

    def test_invalid_corrupted_pdf_raises_invalid_pdf_error(self) -> None:
        """Verify corrupted data raises InvalidPDFError."""
        with self.assertRaises(InvalidPDFError):
            extract_text_from_pdf(b"This is definitely not a PDF file content")

    def test_nonexistent_file_raises_file_not_found(self) -> None:
        """Verify non-existent file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("non_existent_report_12345.pdf")

    def test_unsupported_source_type_raises_invalid_pdf_error(self) -> None:
        """Verify unsupported input type raises InvalidPDFError."""
        with self.assertRaises(InvalidPDFError):
            extract_text_from_pdf(12345)  # type: ignore

    def test_real_pdf_end_to_end_using_synthetic_pdf(self) -> None:
        """End-to-end integration test parsing a real synthetic PDF without external dependencies."""
        # Generate a small 2-page synthetic PDF document in-memory using pypdfium2
        doc = pdfium.PdfDocument.new()
        doc.new_page(300, 200)
        doc.new_page(300, 200)

        pdf_buffer = io.BytesIO()
        doc.save(pdf_buffer)
        pdf_buffer.seek(0)

        # Parse the synthetic PDF
        result = extract_text_from_pdf(pdf_buffer)

        self.assertEqual(result.total_pages, 2)
        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertEqual(result.pages[1].page_number, 2)
        self.assertFalse(result.pages[0].has_text)
        self.assertFalse(result.pages[1].has_text)
        self.assertEqual(result.total_characters, 0)
        self.assertEqual(result.full_text, "")


if __name__ == "__main__":
    unittest.main()
