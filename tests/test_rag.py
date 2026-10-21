"""Unit tests for FinSight RAG document chunking and retrieval pipeline."""

import unittest
from typing import List

from ingestion.pdf_parser import ParsedPage, ParsedPDF
from rag.chunker import DocumentChunker
from rag.pipeline import RAGPipeline
from rag.retriever import (
    BaseRetriever,
    BM25Retriever,
    VectorRetriever,
    jaccard_similarity,
    tokenize,
)
from rag.schema import DocumentChunk, RetrievalResult


class TestDocumentChunker(unittest.TestCase):
    """Test suite for DocumentChunker."""

    def setUp(self) -> None:
        self.chunker = DocumentChunker(chunk_size=200, chunk_overlap=50, min_chunk_size=15)

    def test_chunk_overlap_between_consecutive_chunks(self) -> None:
        """Verify consecutive chunks from a page have overlapping text."""
        long_text = (
            "FinSight Corporation reported net revenue of $500 million for fiscal year 2023. "
            "Cost of goods sold was $280 million resulting in gross profit of $220 million.\n\n"
            "Operating expenses including research and development rose to $110 million. "
            "Sales and marketing expenditure was carefully managed across all regions.\n\n"
            "Net income reached an all-time high of $85 million representing strong growth."
        )
        page = ParsedPage(page_number=3, text=long_text, has_text=True, char_count=len(long_text))
        chunks = self.chunker.chunk_page(page, doc_id="annual_2023")

        self.assertGreater(len(chunks), 1, "Expected text to produce multiple chunks")

        # Verify consecutive chunks have meaningful overlap
        for i in range(len(chunks) - 1):
            curr_text = chunks[i].text
            next_text = chunks[i + 1].text

            # Extract words from the end of current chunk and beginning of next chunk
            curr_words = set(tokenize(curr_text[-80:]))
            next_words = set(tokenize(next_text[:80]))
            common_overlap = curr_words & next_words

            self.assertTrue(
                len(common_overlap) > 0,
                f"Expected overlap between chunk {i} and {i+1}. "
                f"Chunk {i} tail: '{curr_text[-40:]}', Chunk {i+1} head: '{next_text[:40]}'",
            )

    def test_page_attribution_preserved(self) -> None:
        """Verify chunks retain the exact 1-indexed page number from the source ParsedPage."""
        p1 = ParsedPage(
            page_number=1,
            text="Executive Summary: Total revenue exceeded expectations across all business lines.",
            has_text=True,
            char_count=80,
        )
        p2 = ParsedPage(
            page_number=5,
            text="Balance sheet summary: Total assets were $1.2B with $400M in long-term debt.",
            has_text=True,
            char_count=77,
        )
        p3 = ParsedPage(
            page_number=12,
            text="Cash flow statement: Operating cash flow generated was $320M during the year.",
            has_text=True,
            char_count=78,
        )
        pdf = ParsedPDF(total_pages=12, pages=[p1, p2, p3], total_characters=235)

        chunks = self.chunker.chunk_pdf(pdf, doc_id="test_doc")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].chunk_id, "test_doc_p1_c0")

        self.assertEqual(chunks[1].page_number, 5)
        self.assertEqual(chunks[1].chunk_id, "test_doc_p5_c0")

        self.assertEqual(chunks[2].page_number, 12)
        self.assertEqual(chunks[2].chunk_id, "test_doc_p12_c0")

    def test_short_pages_handled_safely(self) -> None:
        """Verify short pages produce exactly one chunk without truncation or errors."""
        short_text = "Item 1. Business Overview."
        page = ParsedPage(page_number=2, text=short_text, has_text=True, char_count=len(short_text))
        chunks = self.chunker.chunk_page(page)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, short_text)
        self.assertEqual(chunks[0].page_number, 2)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_empty_and_blank_pages_produce_no_chunks(self) -> None:
        """Verify blank or whitespace-only pages return empty chunk lists safely."""
        p_empty = ParsedPage(page_number=1, text="", has_text=False, char_count=0)
        p_whitespace = ParsedPage(page_number=2, text="   \n\t  \n  ", has_text=True, char_count=10)

        self.assertEqual(self.chunker.chunk_page(p_empty), [])
        self.assertEqual(self.chunker.chunk_page(p_whitespace), [])

    def test_empty_parsed_pdf_handled_safely(self) -> None:
        """Verify empty PDF or None produces empty list without crashing."""
        self.assertEqual(self.chunker.chunk_pdf(None), [])
        empty_pdf = ParsedPDF(total_pages=0, pages=[], total_characters=0)
        self.assertEqual(self.chunker.chunk_pdf(empty_pdf), [])

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Verify invalid chunk size or overlap parameters raise ValueError."""
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)  # overlap >= size
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=150)  # overlap > size
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=-10, chunk_overlap=5)  # negative size
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=-10)  # negative overlap


class TestBM25Retriever(unittest.TestCase):
    """Test suite for BM25Retriever relevance, deduplication, and edge cases."""

    def setUp(self) -> None:
        self.retriever = BM25Retriever(k1=1.5, b=0.75, dedup_threshold=0.80)
        self.sample_chunks = [
            DocumentChunk(
                chunk_id="chunk_p1",
                page_number=1,
                text="The company achieved record revenue of $450 million with gross margin of 48%.",
                char_count=80,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="chunk_p2",
                page_number=2,
                text="Research and development expenses focused on cloud infrastructure and artificial intelligence.",
                char_count=95,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="chunk_p3",
                page_number=3,
                text="Environmental compliance and sustainability initiatives reduced carbon emissions by 15%.",
                char_count=90,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="chunk_p4",
                page_number=4,
                text="Total debt was $120 million while shareholders' equity was $300 million.",
                char_count=73,
                chunk_index=0,
            ),
        ]
        self.retriever.index(self.sample_chunks)

    def test_retrieval_relevance_accuracy(self) -> None:
        """Verify query returns the most relevant passage with correct page attribution."""
        # Query 1: Revenue & margin
        results = self.retriever.retrieve("What was the revenue and gross margin?", top_k=2)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertEqual(top_result.page_number, 1)
        self.assertIn("revenue", top_result.text.lower())
        self.assertGreater(top_result.score, 0.0)

        # Query 2: Debt and equity
        results_debt = self.retriever.retrieve("total debt and shareholders equity", top_k=1)
        self.assertEqual(len(results_debt), 1)
        self.assertEqual(results_debt[0].page_number, 4)
        self.assertIn("total debt", results_debt[0].text.lower())

        # Query 3: Sustainability
        results_env = self.retriever.retrieve("carbon emissions and environmental compliance", top_k=1)
        self.assertEqual(len(results_env), 1)
        self.assertEqual(results_env[0].page_number, 3)

    def test_irrelevant_query_returns_no_results(self) -> None:
        """Verify queries with zero matching keywords return empty results safely."""
        results = self.retriever.retrieve("quantum astrophysics supernova galaxies", top_k=5)
        self.assertEqual(results, [])

    def test_empty_query_and_empty_retriever(self) -> None:
        """Verify empty/whitespace queries or unindexed retrievers return empty results."""
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   \t  "), [])

        empty_retriever = BM25Retriever()
        self.assertEqual(empty_retriever.retrieve("revenue"), [])
        self.assertEqual(empty_retriever.chunk_count, 0)

    def test_duplicate_chunks_deduplication(self) -> None:
        """Verify near-duplicate or overlapping chunks are deduplicated from results."""
        # Create identical/overlapping chunks
        dup_chunks = [
            DocumentChunk(
                chunk_id="dup_1",
                page_number=1,
                text="Operating cash flow reached $95 million in fiscal year 2023.",
                char_count=60,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="dup_2",
                page_number=1,
                text="Operating cash flow reached $95 million in fiscal year 2023.",  # exact duplicate
                char_count=60,
                chunk_index=1,
            ),
            DocumentChunk(
                chunk_id="other_1",
                page_number=2,
                text="Capital expenditures were $20 million during the period.",
                char_count=57,
                chunk_index=0,
            ),
        ]
        retriever = BM25Retriever(dedup_threshold=0.75)
        retriever.index(dup_chunks)

        results = retriever.retrieve("operating cash flow", top_k=5)
        # Should return exactly 1 operating cash flow chunk, not both dup_1 and dup_2
        self.assertEqual(len(results), 1)
        self.assertIn("operating cash flow", results[0].text.lower())

    def test_retrieval_result_properties(self) -> None:
        """Verify RetrievalResult properties expose chunk attributes cleanly."""
        chunk = self.sample_chunks[0]
        res = RetrievalResult(chunk=chunk, score=1.85)

        self.assertEqual(res.page_number, chunk.page_number)
        self.assertEqual(res.text, chunk.text)
        self.assertEqual(res.chunk_id, chunk.chunk_id)
        self.assertEqual(res.score, 1.85)


class TestModularRetrieverSwapping(unittest.TestCase):
    """Verify that retrieval implementations can be swapped without changing the pipeline interface."""

    def test_custom_mock_retriever(self) -> None:
        """Verify custom retriever implementing BaseRetriever plugs seamlessly into RAGPipeline."""

        class MockKeywordRetriever(BaseRetriever):
            def __init__(self) -> None:
                self._chunks: List[DocumentChunk] = []

            def index(self, chunks: List[DocumentChunk]) -> None:
                self._chunks = list(chunks)

            def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievalResult]:
                q_lower = query.lower()
                matches = [c for c in self._chunks if q_lower in c.text.lower()]
                return [RetrievalResult(chunk=c, score=1.0) for c in matches[:top_k]]

            @property
            def chunk_count(self) -> int:
                return len(self._chunks)

            def clear(self) -> None:
                self._chunks = []

        custom_retriever = MockKeywordRetriever()
        pipeline = RAGPipeline(retriever=custom_retriever)

        p1 = ParsedPage(page_number=1, text="Quarterly dividends were $0.25 per share.", has_text=True, char_count=42)
        pdf = ParsedPDF(total_pages=1, pages=[p1], total_characters=42)

        pipeline.ingest(pdf)
        self.assertEqual(pipeline.chunk_count, 1)

        results = pipeline.search("dividends")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_number, 1)
        self.assertIn("dividends", results[0].text)

    def test_vector_retriever_with_embeddings_function(self) -> None:
        """Verify VectorRetriever indexes and retrieves via pluggable embedding vectors."""

        # Simple deterministic word-hash embedding function for testing
        def mock_embed(texts: List[str]) -> List[List[float]]:
            vocab = ["revenue", "profit", "debt", "cash", "asset"]
            vecs = []
            for t in texts:
                t_lower = t.lower()
                vec = [float(t_lower.count(v)) for v in vocab]
                # Normalize or ensure non-zero
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                vecs.append([x / norm for x in vec])
            return vecs

        vec_retriever = VectorRetriever(embed_fn=mock_embed)
        chunks = [
            DocumentChunk(chunk_id="c1", page_number=1, text="Total revenue was strong.", char_count=25),
            DocumentChunk(chunk_id="c2", page_number=2, text="Total debt was moderate.", char_count=25),
        ]
        vec_retriever.index(chunks)
        self.assertEqual(vec_retriever.chunk_count, 2)

        results = vec_retriever.retrieve("revenue", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_number, 1)

    def test_vector_retriever_without_embed_fn_raises_error(self) -> None:
        """Verify VectorRetriever raises informative ValueError if embed_fn is missing."""
        vec_retriever = VectorRetriever(embed_fn=None)
        chunks = [DocumentChunk(chunk_id="c1", page_number=1, text="Sample text", char_count=11)]
        with self.assertRaises(ValueError):
            vec_retriever.index(chunks)


class TestRAGPipelineEndToEnd(unittest.TestCase):
    """Integration tests for the complete RAG pipeline."""

    def test_full_pipeline_ingest_and_search(self) -> None:
        """Verify end-to-end ingestion and search workflow."""
        pages = [
            ParsedPage(
                page_number=1,
                text="FinSight Overview: The company specializes in enterprise AI analytics and financial intelligence.",
                has_text=True,
                char_count=98,
            ),
            ParsedPage(
                page_number=2,
                text="Financial Highlights: Consolidated revenue grew 24% to $125 million with net profit of $18 million.",
                has_text=True,
                char_count=101,
            ),
            ParsedPage(
                page_number=3,
                text="Risk Factors: Supply chain disruptions and currency exchange fluctuations may adversely impact operations.",
                has_text=True,
                char_count=110,
            ),
        ]
        pdf = ParsedPDF(file_name="report_2023.pdf", total_pages=3, pages=pages, total_characters=309)

        pipeline = RAGPipeline()
        chunks = pipeline.ingest(pdf)

        self.assertGreater(pipeline.chunk_count, 0)
        self.assertEqual(pipeline.indexed_pages, [1, 2, 3])

        # Search for revenue
        results = pipeline.search("consolidated revenue and net profit", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].page_number, 2)
        self.assertIn("revenue", results[0].text.lower())

        # Search for risk
        risk_results = pipeline.search("currency fluctuations and supply chain risks", top_k=1)
        self.assertEqual(len(risk_results), 1)
        self.assertEqual(risk_results[0].page_number, 3)

        # Clear pipeline
        pipeline.clear()
        self.assertEqual(pipeline.chunk_count, 0)
        self.assertEqual(pipeline.search("revenue"), [])


if __name__ == "__main__":
    unittest.main()
