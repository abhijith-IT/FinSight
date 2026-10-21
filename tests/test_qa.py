"""Unit tests for FinSight report-grounded Q&A module."""

import unittest
from unittest.mock import MagicMock

from ingestion.pdf_parser import ParsedPage, ParsedPDF
from rag.pipeline import RAGPipeline
from rag.qa import QAResult, ReportQAEngine
from rag.schema import DocumentChunk, RetrievalResult


class TestReportQAEngine(unittest.TestCase):
    """Test suite for ReportQAEngine with mocked Gemini model."""

    def setUp(self) -> None:
        """Create a populated RAGPipeline and mock Gemini client."""
        pages = [
            ParsedPage(
                page_number=4,
                text="In fiscal year 2023, total revenue reached $850 million, an increase of 15% year-over-year. "
                     "Cost of goods sold was $420 million.",
                has_text=True,
                char_count=138,
            ),
            ParsedPage(
                page_number=7,
                text="Operating cash flow was $320 million for the year. Capital expenditures totaled $80 million, "
                     "primarily invested in cloud computing hardware.",
                has_text=True,
                char_count=154,
            ),
            ParsedPage(
                page_number=14,
                text="Risk Factors: Foreign currency fluctuations and semiconductor supply chain constraints "
                     "may adversely affect our international operating margins.",
                has_text=True,
                char_count=161,
            ),
        ]
        pdf = ParsedPDF(file_name="annual_report_2023.pdf", total_pages=14, pages=pages, total_characters=453)
        self.pipeline = RAGPipeline()
        self.pipeline.ingest(pdf)

        self.mock_client = MagicMock()
        self.qa_engine = ReportQAEngine(
            api_key="mock_key",
            model_name="gemini-3.1-flash-lite",
            client=self.mock_client,
            top_k=3,
        )

    def test_evidence_grounded_answer_with_citations(self) -> None:
        """Verify evidence-grounded answer with accurate [Page X] citations."""
        mock_response = MagicMock()
        mock_response.text = (
            "Total revenue reached $850 million in fiscal year 2023 [Page 4], with cost of goods "
            "sold of $420 million [Page 4]. Capital expenditures were $80 million [Page 7]."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        result = self.qa_engine.answer_question(
            "What was the total revenue and capital expenditure?",
            self.pipeline,
        )

        self.assertIsInstance(result, QAResult)
        self.assertIn("$850 million", result.answer)
        self.assertIn("[Page 4]", result.answer)
        self.assertIn("[Page 7]", result.answer)
        self.assertEqual(result.cited_pages, [4, 7])
        self.assertFalse(result.has_insufficient_evidence)
        self.assertIsNone(result.error)
        self.assertGreater(len(result.sources), 0)

        # Verify Gemini prompt structure
        call_kwargs = self.mock_client.models.generate_content.call_args[1]
        self.assertIn("Question:", call_kwargs["contents"])
        self.assertIn("Retrieved Report Passages:", call_kwargs["contents"])
        self.assertIn("[Page 4]", call_kwargs["contents"])

    def test_citation_validation_filters_unretrieved_invented_pages(self) -> None:
        """Verify that hallucinated citations to unretrieved pages are not validated in cited_pages."""
        mock_response = MagicMock()
        # Model hallucinated [Page 99] which was never in the retrieved passages (only 4 and 7 were retrieved)
        mock_response.text = "Revenue was $850 million [Page 4] and market share grew 10% [Page 99]."
        self.mock_client.models.generate_content.return_value = mock_response

        result = self.qa_engine.answer_question("What was revenue?", self.pipeline)

        # Page 4 is valid and retrieved; Page 99 must be excluded from cited_pages
        self.assertIn(4, result.cited_pages)
        self.assertNotIn(99, result.cited_pages)

    def test_insufficient_evidence_when_model_states_so(self) -> None:
        """Verify has_insufficient_evidence flag when model detects missing facts."""
        mock_response = MagicMock()
        mock_response.text = (
            "The provided report passages do not contain sufficient evidence to answer this question. "
            "CEO compensation details are not reported in the provided text."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        result = self.qa_engine.answer_question("What was the CEO's total salary?", self.pipeline)

        self.assertTrue(result.has_insufficient_evidence)
        self.assertIn("do not contain sufficient evidence", result.answer)

    def test_empty_retrieval_returns_insufficient_evidence_without_llm_call(self) -> None:
        """Verify that a query with zero matching passages returns without calling Gemini."""
        # Query with terms not in any chunk
        result = self.qa_engine.answer_question(
            "quantum teleportation astrophysics supernova",
            self.pipeline,
        )

        self.assertTrue(result.has_insufficient_evidence)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.cited_pages, [])
        self.assertIn("do not contain sufficient evidence", result.answer)
        # Ensure Gemini was never called
        self.mock_client.models.generate_content.assert_not_called()

    def test_gemini_api_error_handled_gracefully(self) -> None:
        """Verify Gemini API failure does not crash the app and returns an informative error result."""
        self.mock_client.models.generate_content.side_effect = Exception("Rate limit exceeded: 429")

        result = self.qa_engine.answer_question("What were the risk factors?", self.pipeline)

        self.assertIsNotNone(result.error)
        self.assertIn("Rate limit exceeded", result.error)
        self.assertIn("Gemini API request failed", result.answer)
        # Sources are still retained so the user has the retrieved context
        self.assertGreater(len(result.sources), 0)

    def test_empty_question_and_empty_pipeline(self) -> None:
        """Verify empty inputs are handled safely without raising exceptions."""
        # Empty question
        res_empty_q = self.qa_engine.answer_question("", self.pipeline)
        self.assertTrue(res_empty_q.has_insufficient_evidence)
        self.assertIn("valid question", res_empty_q.answer)

        # Empty pipeline
        empty_pipe = RAGPipeline()
        res_empty_pipe = self.qa_engine.answer_question("What is revenue?", empty_pipe)
        self.assertTrue(res_empty_pipe.has_insufficient_evidence)
        self.assertIn("not been indexed", res_empty_pipe.answer)


if __name__ == "__main__":
    unittest.main()
