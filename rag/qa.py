"""Report-grounded question answering module for FinSight RAG.

Retrieves relevant document chunks from the RAG pipeline and uses the configured
Gemini model to generate strictly grounded answers with page citations ([Page X]).
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Set

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

import config
from rag.pipeline import RAGPipeline
from rag.schema import RetrievalResult


QA_SYSTEM_INSTRUCTION = """You are FinSight's expert AI financial research assistant.
Your objective is to answer questions about a corporate annual report or financial statement using ONLY the provided text passages.

MANDATORY GROUNDING RULES:
1. STRICT GROUNDING: Answer using ONLY facts directly reported in the provided passages. NEVER use prior knowledge, speculate, or extrapolate facts not present in the text.
2. CITATIONS: Every claim, financial metric, and statement MUST cite the source page number using the format [Page X] (e.g., [Page 4], [Page 12]) corresponding to the page heading where the evidence was found.
3. NEVER INVENT CITATIONS: Only cite page numbers that exist in the provided passages. Never guess or fabricate page numbers.
4. INSUFFICIENT EVIDENCE: If the provided passages do not contain sufficient evidence or information to answer the question, state clearly: "The report does not contain enough information to answer this question." Do not attempt to guess or invent facts.
5. OBJECTIVE & PRECISE: Maintain a concise, objective financial tone. Present numbers, currencies, and scale units accurately as stated in the text.
"""


class QAResult(BaseModel):
    """Structured result returned by the ReportQAEngine."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., description="The user question asked")
    answer: str = Field(..., description="The grounded answer with [Page X] citations")
    sources: List[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved document chunks used as evidence",
    )
    cited_pages: List[int] = Field(
        default_factory=list,
        description="Sorted list of unique 1-indexed page numbers cited in the answer",
    )
    has_insufficient_evidence: bool = Field(
        default=False,
        description="True if the evidence was insufficient to answer the question",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Gemini model name used to generate the answer",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if retrieval or generation failed",
    )


class ReportQAEngine:
    """Answers questions about a financial report grounded strictly in RAG retrieved chunks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
    ) -> None:
        """Initialize the ReportQAEngine.

        Args:
            api_key: Gemini API key. Defaults to config.GEMINI_API_KEY.
            model_name: Primary Gemini model name. Defaults to config.GEMINI_MODEL.
            fallback_model_name: Optional fallback model if primary is unavailable.
            client: Optional pre-configured genai.Client instance.
            top_k: Default number of relevant chunks to retrieve.
            min_score: Minimum relevance score threshold for retrieval.
            max_retries: Maximum retries for transient 503/429 errors.
            base_delay_seconds: Base delay for exponential backoff.
        """
        self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
        self.model_name = model_name or getattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
        self.fallback_model_name = (
            fallback_model_name
            if fallback_model_name is not None
            else getattr(config, "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
        )
        self._client = client
        self.top_k = top_k
        self.min_score = min_score
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds

    @property
    def client(self) -> genai.Client:
        """Get or initialize the google-genai Client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Gemini API key is not configured. Provide api_key or set GEMINI_API_KEY in .env."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def answer_question(
        self,
        question: str,
        pipeline: RAGPipeline,
        top_k: Optional[int] = None,
    ) -> QAResult:
        """Answer a question grounded in passages retrieved from the RAG pipeline.

        Args:
            question: User's question about the report.
            pipeline: RAGPipeline containing indexed report chunks.
            top_k: Optional override for the number of chunks to retrieve.

        Returns:
            QAResult with grounded answer, citations, and source passages.
        """
        # Validate question
        if not question or not question.strip():
            return QAResult(
                question=question or "",
                answer="Please enter a valid question about the report.",
                sources=[],
                cited_pages=[],
                has_insufficient_evidence=True,
            )

        clean_question = question.strip()

        # Validate pipeline
        if pipeline is None or pipeline.chunk_count == 0:
            return QAResult(
                question=clean_question,
                answer="The report has not been indexed or contains no text. Please upload and analyze a report first.",
                sources=[],
                cited_pages=[],
                has_insufficient_evidence=True,
            )

        # Retrieve relevant passages
        effective_top_k = top_k or self.top_k
        try:
            results = pipeline.search(
                query=clean_question,
                top_k=effective_top_k,
                min_score=self.min_score,
            )
        except Exception as ret_err:
            return QAResult(
                question=clean_question,
                answer=f"Retrieval error occurred while searching the report: {ret_err}",
                sources=[],
                cited_pages=[],
                has_insufficient_evidence=True,
                error=str(ret_err),
            )

        # Handle empty retrieval results
        if not results:
            return QAResult(
                question=clean_question,
                answer="The report does not contain enough information to answer this question. (The provided report passages do not contain sufficient evidence.)",
                sources=[],
                cited_pages=[],
                has_insufficient_evidence=True,
            )

        # Build grounded context
        context_blocks = []
        retrieved_pages: Set[int] = set()
        for res in results:
            retrieved_pages.add(res.page_number)
            context_blocks.append(
                f"--- [Page {res.page_number}] (Chunk: {res.chunk_id}) ---\n"
                f"{res.text}"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = (
            f"Question:\n{clean_question}\n\n"
            f"Retrieved Report Passages:\n{context_text}\n\n"
            f"Instructions:\n"
            f"- Answer the question using ONLY the passages above.\n"
            f"- Include [Page X] citations for every statement or number.\n"
            f"- If the passages lack sufficient evidence, state clearly that the report does not contain enough information to answer this question."
        )

        generate_config = types.GenerateContentConfig(
            system_instruction=QA_SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

        # Execute generation with retry and fallback
        raw_answer, model_used, error_msg = self._generate_with_retry_and_fallback(
            prompt=prompt,
            config=generate_config,
        )

        if error_msg:
            return QAResult(
                question=clean_question,
                answer=f"Gemini API request failed: {error_msg}",
                sources=results,
                cited_pages=sorted(retrieved_pages),
                has_insufficient_evidence=False,
                model_used=model_used,
                error=error_msg,
            )

        if not raw_answer.strip():
            raw_answer = "The report does not contain enough information to answer this question. (The provided report passages do not contain sufficient evidence.)"

        # Extract [Page X] citations and reconcile with actual retrieved pages
        cited_matches = re.findall(r"\[Page\s+(\d+)\]", raw_answer, re.IGNORECASE)
        cited_pages = sorted({int(p) for p in cited_matches if int(p) in retrieved_pages})

        # Detect insufficient evidence phrasing
        lower_ans = raw_answer.lower()
        insufficient_evidence = (
            "not contain enough information" in lower_ans
            or "does not contain enough information" in lower_ans
            or "not provide enough information" in lower_ans
            or "does not provide enough information" in lower_ans
            or "insufficient evidence" in lower_ans
            or "not contain sufficient evidence" in lower_ans
            or "does not contain sufficient evidence" in lower_ans
            or "cannot be answered from the provided" in lower_ans
            or "not mentioned in the provided" in lower_ans
            or "not found in the provided" in lower_ans
            or "not reported in the provided" in lower_ans
        )

        return QAResult(
            question=clean_question,
            answer=raw_answer.strip(),
            sources=results,
            cited_pages=cited_pages,
            has_insufficient_evidence=insufficient_evidence,
            model_used=model_used,
        )

    def _generate_with_retry_and_fallback(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Call Gemini model with bounded exponential backoff and fallback model support."""
        models_to_try = [self.model_name]
        if self.fallback_model_name and self.fallback_model_name != self.model_name:
            models_to_try.append(self.fallback_model_name)

        last_error = None

        for m_idx, current_model in enumerate(models_to_try):
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config,
                    )
                    text = getattr(response, "text", "") or ""
                    return text, current_model, None
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_transient = any(
                        term in err_str
                        for term in ["503", "429", "unavailable", "resource_exhausted", "timeout"]
                    )
                    # Non-transient errors (like invalid API key) should not be retried
                    if not is_transient:
                        return "", current_model, str(e)

                    if attempt < self.max_retries:
                        delay = self.base_delay_seconds * (2 ** attempt) + random.uniform(0.1, 0.4)
                        time.sleep(delay)
                        continue

            # If retries exhausted for primary model, proceed to fallback model
            continue

        return "", None, str(last_error) if last_error else "API request failed"
