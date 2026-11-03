"""Financial data extractor for FinSight.

Extracts structured financial figures (Income Statement, Balance Sheet, Cash Flow)
from parsed PDF pages using the google-genai SDK, processing documents in manageable sections.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import random
import re
import time
from typing import Dict, List, Optional, Set, Union

from google import genai
from google.genai import types

import config
from extraction.financial_schema import (
    AnnualFinancialData,
    FinancialExtractionReport,
    FinancialValue,
)
from extraction.validation import (
    clean_and_parse_llm_json,
    sanitize_and_flag_annual_data,
    validate_financial_report,
)
from ingestion.pdf_parser import ParsedPDF, ParsedPage, extract_text_from_pdf

logger = logging.getLogger(__name__)


def is_transient_error(err: Exception) -> bool:
    """Check if an exception represents a transient Gemini error (503, 429, 5xx)."""
    cause = getattr(err, "__cause__", None)
    if cause is not None and isinstance(cause, Exception):
        if is_transient_error(cause):
            return True

    code = getattr(err, "code", None)
    if isinstance(code, int):
        if code in (429, 500, 502, 503, 504):
            return True
        if 400 <= code < 500:
            return False

    status = getattr(err, "status", None)
    if isinstance(status, str):
        status_upper = status.upper()
        if any(
            s in status_upper
            for s in (
                "UNAVAILABLE",
                "RESOURCE_EXHAUSTED",
                "DEADLINE_EXCEEDED",
                "INTERNAL",
                "ABORTED",
            )
        ):
            return True
        if any(
            s in status_upper
            for s in (
                "INVALID_ARGUMENT",
                "PERMISSION_DENIED",
                "NOT_FOUND",
                "UNAUTHENTICATED",
            )
        ):
            return False

    err_str = str(err).lower()
    if any(
        p in err_str
        for p in (
            "api_key",
            "api key",
            "unauthenticated",
            "invalid_argument",
            "permission_denied",
            "not_found",
            "400",
            "401",
            "403",
            "404",
        )
    ):
        return False

    if any(
        t in err_str
        for t in (
            "503",
            "429",
            "500",
            "502",
            "504",
            "unavailable",
            "resource_exhausted",
            "overloaded",
            "high demand",
            "rate limit",
            "too many requests",
            "temporarily",
            "timeout",
            "timed out",
            "service unavailable",
        )
    ):
        return True

    return False


def is_permanent_error(err: Exception) -> bool:
    """Check if an exception is explicitly a permanent error (400, 401, 403, 404)."""
    cause = getattr(err, "__cause__", None)
    if cause is not None and isinstance(cause, Exception):
        if is_permanent_error(cause):
            return True

    code = getattr(err, "code", None)
    if isinstance(code, int) and 400 <= code < 500 and code != 429:
        return True

    status = getattr(err, "status", None)
    if isinstance(status, str):
        status_upper = status.upper()
        if any(
            s in status_upper
            for s in (
                "INVALID_ARGUMENT",
                "PERMISSION_DENIED",
                "NOT_FOUND",
                "UNAUTHENTICATED",
            )
        ):
            return True

    err_str = str(err).lower()
    if any(
        p in err_str
        for p in (
            "api_key",
            "api key",
            "unauthenticated",
            "invalid_argument",
            "permission_denied",
            "not_found",
            "400",
            "401",
            "403",
            "404",
        )
    ):
        return True

    return False


def extract_retry_delay(err: Exception) -> Optional[float]:
    """Extract server-specified retry delay from response headers or error message."""
    cause = getattr(err, "__cause__", None)
    if cause is not None and isinstance(cause, Exception):
        del_cause = extract_retry_delay(cause)
        if del_cause is not None:
            return del_cause

    response = getattr(err, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass

    err_str = str(err)
    match = re.search(
        r"(?:retry\s+after|retry[-_]after|wait)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*s?",
        err_str,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass

    return None


_SECTION_EXTRACTION_CACHE: Dict[str, FinancialExtractionReport] = {}


def clear_section_cache() -> None:
    """Clear the global section extraction cache."""
    _SECTION_EXTRACTION_CACHE.clear()

PRIMARY_STATEMENT_KEYWORDS = [
    # English / US GAAP / IFRS headings
    "balance sheet",
    "statement of financial position",
    "statement of operations",
    "income statement",
    "statement of earnings",
    "profit and loss",
    "statement of profit and loss",
    "statement of income",
    "statement of cash flows",
    "cash flow statement",
    # Indian Companies Act / Reliance-style headings
    "statement of profit & loss",
    "profit & loss account",
    "profit and loss account",
    "standalone statement of profit",
    "consolidated statement of profit",
    "standalone balance sheet",
    "consolidated balance sheet",
    "standalone statement of cash",
    "consolidated statement of cash",
    "annual accounts",
]

CORE_METRIC_KEYWORDS = [
    "cost of goods sold",
    "cost of sales",
    "cost of revenue",
    "cogs",
    "total debt",
    "borrowings",               # Indian balance sheets use "borrowings" not "total debt"
    "total assets",
    "total liabilities",
    "shareholders' equity",
    "stockholders' equity",
    "equity attributable",      # Consolidated reports distinguish NCI vs owners' equity
    "operating activities",
    "investing activities",
    "financing activities",
    "operating expenses",
    "operating profit",
    "operating income",
]

FINANCIAL_STATEMENT_KEYWORDS = [
    "balance sheet",
    "statement of financial position",
    "financial condition",
    "income statement",
    "statement of operations",
    "statement of earnings",
    "profit and loss",
    "p&l",
    "statement of income",
    "statement of cash flows",
    "cash flows",
    "cash flow statement",
    "operating activities",
    "investing activities",
    "financing activities",
    "shareholders' equity",
    "stockholders' equity",
    "equity attributable",
    "total assets",
    "total liabilities",
    "total debt",
    "borrowings",
    "cost of goods sold",
    "cost of sales",
    "cost of revenue",
    "cogs",
    "net income",
    "operating income",
    "operating expenses",
    "revenues",
    # Indian Companies Act terms
    "statement of profit & loss",
    "profit & loss account",
    "profit and loss account",
    "other expenses",
    "finance costs",
    "exceptional items",
]

EXTRACTION_SYSTEM_INSTRUCTION = """You are an expert financial auditor extracting structured financial statement data from annual report pages.
Your objective is to extract accurate figures for each reporting year present in the provided document pages.

MANDATORY RULES:
1. Ground Truth Only: Extract ONLY factual numbers reported in the provided text. Never estimate, derive, or invent values.
2. NEVER Hallucinate: If a metric is not reported for a given year, leave its "value" as null and set "is_unavailable" to true.
3. Comparative Years: When statements show multiple years side-by-side (e.g. 2024, 2023, 2022), extract EACH year into a SEPARATE entry in "annual_reports". Never discard prior-year comparative columns.
4. Source Tracking: For each extracted value, set "source_page" to the exact 1-indexed page number where the number appeared.
5. Currency & Scale: Identify the currency (e.g. USD, EUR, INR) and reporting scale/unit (e.g. millions, billions, thousands, crores) from headers or text.
6. Consolidated vs Standalone: Prefer the CONSOLIDATED financial statements when both are present. Do not mix consolidated and standalone figures.

TARGET METRICS AND MAPPING RULES:

Income Statement:
- revenue: Total revenue, turnover, or net sales AS EXPLICITLY REPORTED. Use the topmost revenue/turnover line.
- cost_of_goods_sold: ONLY populate if the source explicitly reports "Cost of goods sold", "Cost of sales", or "Cost of revenue". NEVER substitute "purchases of stock-in-trade", "changes in inventories", or any other sub-line.
- operating_expenses: ONLY populate if the source reports a line explicitly labelled "Total operating expenses", "Operating expenses", or a clear aggregate OPEX line. NEVER use "Other expenses" (a residual line in Indian P&L formats) as a substitute for this field. If no explicit OPEX aggregate line exists, leave as null.
- operating_profit: The line explicitly labelled "Operating profit", "Profit from operations", "Operating income", or "EBIT" on the face of the financial statement. NEVER substitute EBITDA (Earnings Before Interest, Tax, Depreciation, and Amortisation) for operating profit unless the source document explicitly states they are equivalent. EBITDA ≠ EBIT.
- net_profit: "Profit for the year/period", "Net income", "Net earnings", or "Profit after tax" attributable to the reporting entity.

Balance Sheet:
- total_assets: "Total assets" as the bottom-line asset figure.
- total_liabilities: "Total liabilities" as the bottom-line liability figure.
- total_debt: GROSS BORROWINGS ONLY — short-term borrowings + long-term borrowings as reported under liabilities. NEVER use net debt (borrowings minus cash). NEVER substitute total_liabilities for total_debt. Extract ONLY if the source reports "Total borrowings", "Total debt", or you can identify the borrowings sub-totals under financial liabilities.
- equity: For CONSOLIDATED statements, use "Equity attributable to owners of the parent" (excluding Non-Controlling Interest / minority interest) if that line is reported separately. If only total equity is reported, use that and note it. NEVER use retained earnings alone.

Cash Flow Statement:
- operating_cash_flow: "Net cash from (used in) operating activities" — the net figure after working capital adjustments and taxes, not the profit before adjustments.
- capital_expenditure: "Purchase of property, plant and equipment", "Capital expenditures", "Additions to fixed assets", or "Payments for capital expenditure" — typically shown as a negative number under investing activities.

PRECISION RULES:
7. Explicit Debt vs Liabilities: Extract "total_debt" ONLY if explicitly reported as total debt or borrowings. NEVER substitute total_liabilities for total_debt.
8. Explicit COGS: Extract "cost_of_goods_sold" ONLY if explicitly reported (or reported as cost of sales/cost of revenue). NEVER calculate or infer missing COGS.
9. No EBITDA → Operating Profit substitution: If a source only reports EBITDA but not operating profit, leave operating_profit as null. Do not map EBITDA to operating_profit.
10. No Other Expenses → Operating Expenses substitution: "Other expenses" in Indian accounting formats is a residual catch-all (not total OPEX). Leave operating_expenses as null if no explicit OPEX aggregate line exists.
11. Equity precision: In consolidated statements, prefer equity attributable to owners of the parent over total equity (which includes NCI). Record which figure was used in extraction_notes.
"""


class ExtractionError(Exception):
    """Raised when financial data extraction fails."""
    pass


class FinancialExtractor:
    """Extracts structured financial data from financial PDF reports using Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
        max_pages_per_section: int = 4,
        max_chars_per_section: int = 15000,
        max_sections_to_extract: int = 5,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        max_consecutive_failures: int = 2,
    ) -> None:
        """Initialize the extractor.

        Args:
            api_key: Gemini API key. Defaults to config.GEMINI_API_KEY.
            model_name: Primary Gemini model name. Defaults to config.GEMINI_MODEL.
            fallback_model_name: Optional fallback Gemini model name (e.g. gemini-3.5-flash-lite).
            client: Optional pre-configured genai.Client instance (useful for testing).
            max_pages_per_section: Maximum pages per request chunk.
            max_chars_per_section: Maximum characters per request chunk.
            max_sections_to_extract: Maximum candidate sections to send to Gemini.
            max_retries: Maximum number of retries for transient errors.
            base_delay: Base exponential backoff delay in seconds.
            max_delay: Maximum delay ceiling for backoff in seconds.
            max_consecutive_failures: Number of consecutive section failures before pausing extraction.
        """
        self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
        self.model_name = model_name or getattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
        self.fallback_model_name = fallback_model_name
        self._client = client
        self.max_pages_per_section = max_pages_per_section
        self.max_chars_per_section = max_chars_per_section
        self.max_sections_to_extract = max_sections_to_extract
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_consecutive_failures = max_consecutive_failures
        self._cache: Dict[str, FinancialExtractionReport] = {}
        self._section_cache: Dict[str, FinancialExtractionReport] = {}

    @property
    def client(self) -> genai.Client:
        """Get or initialize the google-genai Client."""
        if self._client is None:
            if not self.api_key:
                raise ExtractionError(
                    "Gemini API key is not configured. Set GEMINI_API_KEY in .env/config.py "
                    "or pass api_key to FinancialExtractor."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _compute_section_cache_key(self, section_pages: List[ParsedPage]) -> str:
        """Compute a deterministic hash for a section based on page numbers and text."""
        raw = "||".join(f"{p.page_number}:{p.text}" for p in section_pages)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _calculate_backoff_delay(self, attempt: int, err: Exception) -> float:
        """Calculate exponential backoff delay with jitter, respecting server delay if present."""
        server_delay = extract_retry_delay(err)
        if server_delay is not None and server_delay > 0:
            return min(server_delay, self.max_delay)

        # Exponential backoff: base_delay * (2 ** attempt) + random jitter between 0.1 and 1.0
        exp_delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 1.0)
        return min(exp_delay + jitter, self.max_delay)

    def partition_pages(self, parsed_pdf: ParsedPDF) -> List[List[ParsedPage]]:
        """Partition a parsed PDF into manageable sections.

        Prioritizes pages with financial keywords if document is large,
        and groups them into bounded page chunks.
        """
        non_empty_pages = [p for p in parsed_pdf.pages if p.has_text and p.text.strip()]
        if not non_empty_pages:
            return []

        # If document is small enough, return as single section
        if len(non_empty_pages) <= self.max_pages_per_section:
            total_chars = sum(len(p.text) for p in non_empty_pages)
            if total_chars <= self.max_chars_per_section:
                return [non_empty_pages]

        # Prioritize primary financial statement pages (Balance Sheet, Income Statement, Cash Flows)
        primary_pages: Set[int] = set()
        for page in non_empty_pages:
            lower_text = page.text.lower()
            if any(kw in lower_text for kw in PRIMARY_STATEMENT_KEYWORDS):
                primary_pages.add(page.page_number)

        relevant_page_nums: Set[int] = set()
        if primary_pages:
            for p_num in primary_pages:
                relevant_page_nums.add(p_num)
                # Include adjacent pages for multi-page tables/notes
                relevant_page_nums.add(p_num - 1)
                relevant_page_nums.add(p_num + 1)

            # Include secondary tables that contain multiple core financial metrics and numbers
            for page in non_empty_pages:
                if page.page_number in relevant_page_nums:
                    continue
                lower_text = page.text.lower()
                metric_matches = sum(1 for kw in CORE_METRIC_KEYWORDS if kw in lower_text)
                digit_count = sum(1 for c in page.text if c.isdigit())
                if metric_matches >= 2 and digit_count >= 15:
                    relevant_page_nums.add(page.page_number)
        else:
            # Fallback to general financial keywords if no primary statement headers matched
            for page in non_empty_pages:
                lower_text = page.text.lower()
                if any(kw in lower_text for kw in FINANCIAL_STATEMENT_KEYWORDS):
                    relevant_page_nums.add(page.page_number)
                    relevant_page_nums.add(page.page_number - 1)
                    relevant_page_nums.add(page.page_number + 1)

        # Filter candidate pages based on relevant page numbers
        candidate_pages = [
            p for p in non_empty_pages if p.page_number in relevant_page_nums
        ]

        # Fallback to all non-empty pages if no keywords matched
        if not candidate_pages:
            candidate_pages = non_empty_pages

        # Chunk candidate pages respecting max_pages and max_chars
        sections: List[List[ParsedPage]] = []
        current_chunk: List[ParsedPage] = []
        current_chars = 0

        for page in candidate_pages:
            page_len = len(page.text)
            if current_chunk and (
                len(current_chunk) >= self.max_pages_per_section
                or (current_chars + page_len) > self.max_chars_per_section
            ):
                sections.append(current_chunk)
                current_chunk = [page]
                current_chars = page_len
            else:
                current_chunk.append(page)
                current_chars += page_len

        if current_chunk:
            sections.append(current_chunk)

        return sections

    def _execute_model_with_retries(
        self,
        model: str,
        prompt: str,
        generate_config: types.GenerateContentConfig,
    ) -> FinancialExtractionReport:
        """Execute extraction with a specified model applying bounded retries and exponential backoff."""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=generate_config,
                )
                raw_text = getattr(response, "text", "") or ""
                parsed_dict = clean_and_parse_llm_json(raw_text)
                return FinancialExtractionReport.model_validate(parsed_dict)

            except Exception as api_err:
                last_err = api_err

                # Permanent error -> fail immediately without retries
                if is_permanent_error(api_err):
                    logger.error(f"Permanent error with model '{model}': {api_err}")
                    raise ExtractionError(f"Gemini API permanent error: {api_err}") from api_err

                # Non-transient error -> fail immediately without retries
                if not is_transient_error(api_err):
                    logger.error(f"Non-transient error with model '{model}': {api_err}")
                    raise ExtractionError(f"Extraction failed: {api_err}") from api_err

                # Transient error (503, 429, 5xx) -> retry if attempts remain
                if attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt, api_err)
                    logger.warning(
                        f"Transient Gemini API error ({api_err}) on model '{model}' attempt {attempt + 1}/{self.max_retries + 1}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Exceeded max retries ({self.max_retries}) on model '{model}' with transient error: {api_err}"
                    )
                    raise api_err

        raise ExtractionError(f"Gemini API request failed on model '{model}': {last_err}")

    def extract_section(self, section_pages: List[ParsedPage]) -> FinancialExtractionReport:
        """Call Gemini to extract structured financial data from a single section.

        Attempts primary model with bounded exponential backoff retries.
        If primary model fails due to transient availability or quota exhaustion (503/429),
        switches to the fallback model if configured.
        Caches successful section reports to avoid repeating LLM calls on retries.

        Args:
            section_pages: Batch of pages to extract from.

        Returns:
            FinancialExtractionReport for this section.
        """
        sec_key = self._compute_section_cache_key(section_pages)
        if sec_key in self._section_cache:
            return self._section_cache[sec_key].model_copy(deep=True)
        if sec_key in _SECTION_EXTRACTION_CACHE:
            cached = _SECTION_EXTRACTION_CACHE[sec_key].model_copy(deep=True)
            self._section_cache[sec_key] = cached
            return cached

        section_text_blocks = []
        for p in section_pages:
            section_text_blocks.append(f"--- PAGE {p.page_number} ---\n{p.text}")
        section_content = "\n\n".join(section_text_blocks)

        prompt = (
            f"Extract all reported financial statements from the following pages.\n\n"
            f"{section_content}"
        )

        generate_config = types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=FinancialExtractionReport,
            temperature=0.0,
        )

        # 1. Attempt with primary model
        primary_model = self.model_name
        try:
            report = self._execute_model_with_retries(
                model=primary_model,
                prompt=prompt,
                generate_config=generate_config,
            )
            # Cache successful section extraction
            self._section_cache[sec_key] = report
            _SECTION_EXTRACTION_CACHE[sec_key] = report
            return report

        except ExtractionError:
            # Permanent or non-transient error -> fail immediately, do not switch to fallback
            raise

        except Exception as primary_err:
            # Primary model exhausted retries with transient error (503/429)
            logger.warning(
                f"Primary model '{primary_model}' failed after retries: {primary_err}"
            )

            # Check if fallback model is configured and distinct from primary
            if not self.fallback_model_name or self.fallback_model_name == primary_model:
                raise ExtractionError(
                    f"Gemini API request failed after {self.max_retries} retries: {primary_err}"
                ) from primary_err

            # 2. Switch to fallback model
            fallback_model = self.fallback_model_name
            logger.info(
                f"Switching to fallback model '{fallback_model}'..."
            )

            try:
                report = self._execute_model_with_retries(
                    model=fallback_model,
                    prompt=prompt,
                    generate_config=generate_config,
                )
                note = (
                    f"Section extracted using fallback model '{fallback_model}' "
                    f"after primary model '{primary_model}' encountered transient errors."
                )
                report.extraction_notes.append(note)

                # Cache successful section extraction
                self._section_cache[sec_key] = report
                _SECTION_EXTRACTION_CACHE[sec_key] = report
                return report

            except ExtractionError:
                # Permanent or non-transient error on fallback
                raise

            except Exception as fallback_err:
                logger.error(
                    f"Both primary model '{primary_model}' and fallback model '{fallback_model}' failed. "
                    f"Primary: {primary_err}; Fallback: {fallback_err}"
                )
                raise ExtractionError(
                    f"Both primary model '{primary_model}' and fallback model '{fallback_model}' failed. "
                    f"Primary: {primary_err}; Fallback: {fallback_err}"
                ) from fallback_err

    def _merge_metric(self, target: FinancialValue, incoming: FinancialValue) -> None:
        """Merge an incoming metric value into target if incoming is non-null."""
        if incoming.value is not None:
            if target.value is None or target.is_unavailable:
                target.value = incoming.value
                target.source_page = incoming.source_page
                target.is_unavailable = False

    def _merge_annual_data(
        self,
        base: AnnualFinancialData,
        incoming: AnnualFinancialData,
    ) -> None:
        """Merge incoming annual data into base record for the same year."""
        if not base.currency and incoming.currency:
            base.currency = incoming.currency
        if not base.unit and incoming.unit:
            base.unit = incoming.unit

        # Merge Income Statement
        self._merge_metric(base.income_statement.revenue, incoming.income_statement.revenue)
        self._merge_metric(
            base.income_statement.cost_of_goods_sold,
            incoming.income_statement.cost_of_goods_sold,
        )
        self._merge_metric(
            base.income_statement.operating_expenses,
            incoming.income_statement.operating_expenses,
        )
        self._merge_metric(
            base.income_statement.operating_profit,
            incoming.income_statement.operating_profit,
        )
        self._merge_metric(
            base.income_statement.net_profit,
            incoming.income_statement.net_profit,
        )

        # Merge Balance Sheet
        self._merge_metric(base.balance_sheet.total_assets, incoming.balance_sheet.total_assets)
        self._merge_metric(
            base.balance_sheet.total_liabilities,
            incoming.balance_sheet.total_liabilities,
        )
        self._merge_metric(
            base.balance_sheet.total_debt,
            incoming.balance_sheet.total_debt,
        )
        self._merge_metric(base.balance_sheet.equity, incoming.balance_sheet.equity)

        # Merge Cash Flow Statement
        self._merge_metric(
            base.cash_flow_statement.operating_cash_flow,
            incoming.cash_flow_statement.operating_cash_flow,
        )
        self._merge_metric(
            base.cash_flow_statement.capital_expenditure,
            incoming.cash_flow_statement.capital_expenditure,
        )

        # Combine source pages
        combined_pages = set(base.source_pages) | set(incoming.source_pages)
        base.source_pages = sorted(list(combined_pages))

    def _is_extraction_complete(self, annual_map: Dict[int, AnnualFinancialData]) -> bool:
        """Check if all standard financial metrics have been extracted across all recorded years."""
        if not annual_map:
            return False
        for ann in annual_map.values():
            inc = ann.income_statement
            bs = ann.balance_sheet
            cf = ann.cash_flow_statement
            if (
                inc.revenue.value is None
                or inc.cost_of_goods_sold.value is None
                or inc.operating_profit.value is None
                or inc.net_profit.value is None
                or bs.total_assets.value is None
                or bs.total_liabilities.value is None
                or bs.total_debt.value is None
                or bs.equity.value is None
                or cf.operating_cash_flow.value is None
                or cf.capital_expenditure.value is None
            ):
                return False
        return True

    def extract(
        self,
        source: Union[ParsedPDF, str, Path, bytes],
    ) -> FinancialExtractionReport:
        """Extract structured financial data from a PDF document.

        Args:
            source: ParsedPDF instance, or file path/bytes to parse.

        Returns:
            FinancialExtractionReport: Consolidated multi-year financial data.
        """
        if isinstance(source, ParsedPDF):
            parsed_pdf = source
        else:
            parsed_pdf = extract_text_from_pdf(source)

        # Check internal instance cache to avoid redundant extraction on same parsed document
        # Only return cached report if it was complete
        cache_key = hashlib.sha256(parsed_pdf.full_text.encode("utf-8")).hexdigest()
        if hasattr(self, "_cache") and cache_key in self._cache:
            cached_report = self._cache[cache_key]
            if not getattr(cached_report, "is_incomplete", False):
                return cached_report.model_copy(deep=True)

        sections = self.partition_pages(parsed_pdf)
        if not sections:
            return FinancialExtractionReport(
                company_name=None,
                annual_reports=[],
                extraction_notes=["Document contained no extractable text pages."],
            )

        company_name: Optional[str] = None
        annual_map: Dict[int, AnnualFinancialData] = {}
        extraction_notes: List[str] = []
        failed_sections: List[int] = []
        unprocessed_sections: List[int] = []

        # Bound active sections to prevent runaway Gemini API calls on large reports
        active_sections = sections[: self.max_sections_to_extract]
        consecutive_transient_failures = 0

        for idx, section in enumerate(active_sections, start=1):
            page_numbers = [p.page_number for p in section]

            try:
                section_report = self.extract_section(section)
                consecutive_transient_failures = 0  # Reset counter on success

                if not company_name and section_report.company_name:
                    company_name = section_report.company_name

                for incoming_annual in section_report.annual_reports:
                    year = incoming_annual.year
                    if year not in annual_map:
                        annual_map[year] = incoming_annual
                    else:
                        self._merge_annual_data(annual_map[year], incoming_annual)

                # Early stopping: if all standard financial statement metrics have been extracted for all years,
                # stop processing remaining sections to avoid redundant LLM calls.
                if self._is_extraction_complete(annual_map):
                    logger.info("Early stopping: all standard financial statement metrics populated.")
                    break

            except Exception as sec_err:
                orig_err = getattr(sec_err, "__cause__", None) or sec_err
                if is_permanent_error(orig_err):
                    # Immediate failure on permanent errors (e.g. invalid API key)
                    raise sec_err

                failed_sections.append(idx)
                if is_transient_error(orig_err):
                    consecutive_transient_failures += 1
                else:
                    consecutive_transient_failures = 0

                msg = f"Section {idx} (pages {page_numbers}) extraction note: {sec_err}"
                logger.warning(msg)
                extraction_notes.append(msg)

                # Stop repeated failures: pause further Gemini requests if consecutive transient failures reached limit
                if consecutive_transient_failures >= self.max_consecutive_failures:
                    pause_msg = (
                        f"Extraction paused: {consecutive_transient_failures} consecutive sections failed "
                        f"with transient availability errors. Stopping further requests to avoid overloading the model."
                    )
                    logger.warning(pause_msg)
                    extraction_notes.append(pause_msg)

                    # Mark remaining sections as unprocessed
                    for rem_idx in range(idx + 1, len(active_sections) + 1):
                        unprocessed_sections.append(rem_idx)
                        rem_pages = [p.page_number for p in active_sections[rem_idx - 1]]
                        extraction_notes.append(
                            f"Section {rem_idx} (pages {rem_pages}) skipped: extraction paused due to model unavailability."
                        )
                    break

        # Sort years chronologically and sanitize
        sorted_annuals = [annual_map[y] for y in sorted(annual_map.keys())]
        for ann in sorted_annuals:
            sanitize_and_flag_annual_data(ann)

        is_incomplete = bool(failed_sections or unprocessed_sections)

        consolidated_report = FinancialExtractionReport(
            company_name=company_name,
            annual_reports=sorted_annuals,
            extraction_notes=extraction_notes,
            is_incomplete=is_incomplete,
            failed_sections=failed_sections,
            unprocessed_sections=unprocessed_sections,
        )

        # Run validation and append findings to notes
        val_summary = validate_financial_report(consolidated_report)
        for issue in val_summary.issues:
            if issue.severity in ("WARNING", "ERROR"):
                consolidated_report.extraction_notes.append(
                    f"[{issue.severity}] {issue.field} (Year {issue.year}): {issue.message}"
                )

        if not hasattr(self, "_cache"):
            self._cache = {}
        # Only cache complete reports at document level to allow recovery on retry
        if not is_incomplete:
            self._cache[cache_key] = consolidated_report

        return consolidated_report
