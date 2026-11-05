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
            fallback_model_name=None,
            client=self.mock_client,
            top_k=3,
            max_retries=1,
            base_delay_seconds=0.01,
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

    def test_transient_error_retry_and_fallback_model(self) -> None:
        """Verify engine retries transient 503 and switches to fallback model."""
        mock_resp = MagicMock()
        mock_resp.text = "Grounded answer from fallback model [Page 4]."

        def generate_mock(model, contents, config):
            if model == "gemini-3.8-flash":
                raise Exception("503 Service Unavailable")
            return mock_resp

        self.mock_client.models.generate_content.side_effect = generate_mock

        qa = ReportQAEngine(
            api_key="mock_key",
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            client=self.mock_client,
            top_k=2,
            max_retries=1,
            base_delay_seconds=0.01,
        )

        result = qa.answer_question("What was revenue?", self.pipeline)
        self.assertFalse(result.has_insufficient_evidence)
        self.assertIn("fallback model", result.answer)
        self.assertEqual(result.model_used, "gemini-3.5-flash-lite")
        self.assertIn(4, result.cited_pages)

    def test_unsupported_out_of_scope_question_with_retrieved_evidence(self) -> None:
        """Verify model returns insufficient evidence when question cannot be answered from passages."""
        mock_resp = MagicMock()
        mock_resp.text = "The report does not contain enough information to answer this question."
        self.mock_client.models.generate_content.return_value = mock_resp

        result = self.qa_engine.answer_question("What was revenue and the CEO's patent number?", self.pipeline)
        self.assertTrue(result.has_insufficient_evidence)
        self.assertIn("not contain enough information", result.answer)

    def test_non_transient_error_fails_immediately_without_retries(self) -> None:
        """Verify non-transient error like invalid API key does not trigger wasteful retries."""
        self.mock_client.models.generate_content.side_effect = Exception("400 Bad Request: Invalid API key")
        result = self.qa_engine.answer_question("What was revenue?", self.pipeline)

        self.assertIsNotNone(result.error)
        self.assertIn("Invalid API key", result.error)
        self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

    def test_consolidated_revenue_distinguished_and_prioritized(self) -> None:
        """Verify consolidated figures are prioritized and distinguished from standalone."""
        pages = [
            ParsedPage(
                page_number=320,
                text="Standalone Statement of Profit and Loss: Revenue from operations was $2,910 million for FY2025.",
                has_text=True,
                char_count=100,
            ),
            ParsedPage(
                page_number=410,
                text="Consolidated Statement of Profit and Loss for the year ended March 31, 2025. "
                     "Revenue from operations was $9,800 million compared to $8,900 million in prior year.",
                has_text=True,
                char_count=170,
            ),
        ]
        pdf = ParsedPDF(file_name="report_multi_statement.pdf", total_pages=450, pages=pages, total_characters=270)
        pipe = RAGPipeline()
        pipe.ingest(pdf)

        mock_resp = MagicMock()
        mock_resp.text = (
            "Consolidated revenue from operations for FY2025 was $9,800 million [Page 410]. "
            "For reference, standalone revenue from operations was $2,910 million [Page 320]."
        )
        self.mock_client.models.generate_content.return_value = mock_resp

        result = self.qa_engine.answer_question("What was the revenue from operations in FY2025?", pipe)

        self.assertFalse(result.has_insufficient_evidence)
        self.assertIn("$9,800 million", result.answer)
        self.assertIn("[Page 410]", result.answer)
        self.assertEqual(result.cited_pages, [320, 410])
        # Verify retrieved top source is Consolidated (Page 410)
        self.assertEqual(result.sources[0].page_number, 410)

        # Verify prompt instructed consolidated priority
        call_kwargs = self.mock_client.models.generate_content.call_args[1]
        self.assertIn("CONSOLIDATED PRIORITY", call_kwargs["contents"])
        self.assertIn("EXACT METRIC", call_kwargs["contents"])

    def test_system_prompt_contains_mandatory_financial_grounding_rules(self) -> None:
        """Verify QA_SYSTEM_INSTRUCTION includes mandatory rules for consolidated vs standalone and exact metrics."""
        from rag.qa import QA_SYSTEM_INSTRUCTION
        self.assertIn("CONSOLIDATED VS. STANDALONE & SEGMENTS", QA_SYSTEM_INSTRUCTION)
        self.assertIn("EXACT METRIC SPECIFICITY", QA_SYSTEM_INSTRUCTION)
        self.assertIn("RISK FACTORS & MANAGEMENT DISCLOSURES", QA_SYSTEM_INSTRUCTION)
        self.assertIn("INSUFFICIENT EVIDENCE & AMBIGUITY", QA_SYSTEM_INSTRUCTION)

    def test_financial_risks_qa_retrieves_erm_and_answers(self) -> None:
        """Verify questions about financial risks retrieve ERM sections and generate citations."""
        pages = [
            ParsedPage(
                page_number=10,
                text="The company achieved major financial success with strong financial discipline and capital allocation.",
                has_text=True,
                char_count=105,
            ),
            ParsedPage(
                page_number=45,
                text="Enterprise Risk Management: The Company is exposed to financial risks including market risk, "
                     "credit risk, and liquidity risk. Management manages exposure through active hedging.",
                has_text=True,
                char_count=175,
            ),
        ]
        pdf = ParsedPDF(file_name="report_erm.pdf", total_pages=50, pages=pages, total_characters=280)
        pipe = RAGPipeline()
        pipe.ingest(pdf)

        mock_resp = MagicMock()
        mock_resp.text = (
            "The major financial risks include market risk, credit risk, and liquidity risk [Page 45]. "
            "These are monitored under the Enterprise Risk Management framework [Page 45]."
        )
        self.mock_client.models.generate_content.return_value = mock_resp

        result = self.qa_engine.answer_question("What are the major financial risks?", pipe)

        self.assertFalse(result.has_insufficient_evidence)
        self.assertIn("market risk", result.answer)
        self.assertEqual(result.cited_pages, [45])
        # Page 45 (ERM) was prioritized over Page 10 (generic financial)
        self.assertEqual(result.sources[0].page_number, 45)

    def test_segment_vs_group_metric_resolution_prioritizes_consolidated_total(self) -> None:
        """Regression test: General company revenue query prioritizes consolidated total over individual segments."""
        pages = [
            ParsedPage(
                page_number=12,
                text="Management Discussion and Analysis - Business Overview: "
                     "Segment Alpha revenue from operations was $3,200 million for the year.",
                has_text=True,
                char_count=130,
            ),
            ParsedPage(
                page_number=88,
                text="Consolidated Statement of Profit and Loss for the year ended December 31, 2025. "
                     "Revenue from operations was $11,500 million compared to $10,100 million in prior year.",
                has_text=True,
                char_count=175,
            ),
            ParsedPage(
                page_number=105,
                text="Note 30: Segment Information. Segment Alpha revenue: $3,200 million; Segment Beta revenue: $8,300 million. "
                     "Total consolidated revenue from operations was $11,500 million.",
                has_text=True,
                char_count=180,
            ),
        ]
        pdf = ParsedPDF(file_name="report_segments.pdf", total_pages=120, pages=pages, total_characters=485)
        pipe = RAGPipeline()
        pipe.ingest(pdf)

        mock_resp = MagicMock()
        mock_resp.text = (
            "Consolidated revenue from operations for FY2025 was $11,500 million [Page 88]. "
            "Individual segment contributions included Segment Alpha at $3,200 million [Page 12] "
            "and Segment Beta at $8,300 million [Page 105]."
        )
        self.mock_client.models.generate_content.return_value = mock_resp

        result = self.qa_engine.answer_question("What was the revenue from operations in FY2025?", pipe)

        self.assertFalse(result.has_insufficient_evidence)
        self.assertIn("$11,500 million", result.answer)
        self.assertIn("[Page 88]", result.answer)
        self.assertEqual(result.cited_pages, [12, 88, 105])
        # Top retrieved source must be the Consolidated Statement of Profit and Loss (Page 88)
        self.assertEqual(result.sources[0].page_number, 88)

        # Verify prompt instructions explicitly enforce segment vs total rules
        call_kwargs = self.mock_client.models.generate_content.call_args[1]
        prompt_content = call_kwargs["contents"]
        self.assertIn("CONSOLIDATED PRIORITY & SEGMENTS", prompt_content)
        self.assertIn("TOTAL figure", prompt_content)
        self.assertIn("Segment values must NEVER be reported as the company-wide metric", prompt_content)

    def test_segment_specific_question_returns_labeled_segment_value(self) -> None:
        """Verify user asking specifically for a segment metric receives the labeled segment value."""
        pages = [
            ParsedPage(
                page_number=12,
                text="Management Discussion and Analysis - Business Overview: "
                     "Segment Alpha revenue from operations was $3,200 million for the year.",
                has_text=True,
                char_count=130,
            ),
            ParsedPage(
                page_number=88,
                text="Consolidated Statement of Profit and Loss: Revenue from operations was $11,500 million.",
                has_text=True,
                char_count=98,
            ),
        ]
        pdf = ParsedPDF(file_name="report_segment_alpha.pdf", total_pages=100, pages=pages, total_characters=228)
        pipe = RAGPipeline()
        pipe.ingest(pdf)

        mock_resp = MagicMock()
        mock_resp.text = "Segment Alpha revenue from operations was $3,200 million [Page 12]."
        self.mock_client.models.generate_content.return_value = mock_resp

        result = self.qa_engine.answer_question("What was Segment Alpha revenue from operations?", pipe)

        self.assertFalse(result.has_insufficient_evidence)
        self.assertIn("$3,200 million", result.answer)
        self.assertIn("Segment Alpha", result.answer)
        self.assertEqual(result.cited_pages, [12])

    def test_multi_segment_table_with_total_row_uses_total_metric(self) -> None:
        """Verify a disaggregated segment table with a Total row extracts the total for enterprise revenue."""
        pages = [
            ParsedPage(
                page_number=50,
                text="Table 4: Disaggregated Revenue by Operating Segment\n"
                     "Segment Hardware: $400 million\n"
                     "Segment Services: $600 million\n"
                     "Total Revenue from Operations: $1,000 million",
                has_text=True,
                char_count=170,
            ),
        ]
        pdf = ParsedPDF(file_name="report_disaggregated.pdf", total_pages=60, pages=pages, total_characters=170)
        pipe = RAGPipeline()
        pipe.ingest(pdf)

        mock_resp = MagicMock()
        mock_resp.text = "Total revenue from operations across all segments was $1,000 million [Page 50]."
        self.mock_client.models.generate_content.return_value = mock_resp

        result = self.qa_engine.answer_question("What was the total revenue from operations?", pipe)

        self.assertFalse(result.has_insufficient_evidence)
        self.assertIn("$1,000 million", result.answer)
        self.assertEqual(result.cited_pages, [50])


if __name__ == "__main__":
    unittest.main()
