"""Financial data extractor for FinSight.

Extracts structured financial figures (Income Statement, Balance Sheet, Cash Flow)
from parsed PDF pages using the google-genai SDK, processing documents in manageable sections.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Union
from pathlib import Path

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
    "total assets",
    "total liabilities",
    "total debt",
    "cost of goods sold",
    "cost of sales",
    "cost of revenue",
    "cogs",
    "net income",
    "operating income",
    "operating expenses",
    "revenues",
]

EXTRACTION_SYSTEM_INSTRUCTION = """You are an expert financial auditor extracting structured financial statement data.
Your objective is to extract accurate figures for each reporting year present in the provided document pages.

MANDATORY RULES:
1. Ground Truth Only: Extract ONLY factual numbers reported in the provided text.
2. NEVER Hallucinate: If a metric is not reported for a given year, leave its "value" as null and set "is_unavailable" to true. Do NOT estimate or derive values.
3. Comparative Years: When statements show multiple years side-by-side (e.g. 2024, 2023, 2022), extract each year into a separate entry in "annual_reports".
4. Source Tracking: For each extracted value, set "source_page" to the exact 1-indexed page number where the number appeared.
5. Currency & Scale: Identify the currency (e.g. USD, EUR, INR) and reporting scale/unit (e.g. millions, billions, thousands) from headers or text.
6. Target Metrics:
   - Income Statement: revenue, cost_of_goods_sold, operating_expenses, operating_profit, net_profit
   - Balance Sheet: total_assets, total_liabilities, total_debt, equity
   - Cash Flow Statement: operating_cash_flow, capital_expenditure
7. Explicit Debt vs Liabilities: Extract "total_debt" ONLY if explicitly reported as total debt or borrowings. NEVER substitute total_liabilities for total_debt.
8. Explicit COGS: Extract "cost_of_goods_sold" ONLY if explicitly reported (or reported as cost of sales/cost of revenue). NEVER calculate or infer missing COGS.
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
        client: Optional[genai.Client] = None,
        max_pages_per_section: int = 4,
        max_chars_per_section: int = 15000,
    ) -> None:
        """Initialize the extractor.

        Args:
            api_key: Gemini API key. Defaults to config.GEMINI_API_KEY.
            model_name: Gemini model name. Defaults to config.GEMINI_MODEL.
            client: Optional pre-configured genai.Client instance (useful for testing).
            max_pages_per_section: Maximum pages per request chunk.
            max_chars_per_section: Maximum characters per request chunk.
        """
        self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
        self.model_name = model_name or getattr(config, "GEMINI_MODEL", "gemini-3.1-flash-lite")
        self._client = client
        self.max_pages_per_section = max_pages_per_section
        self.max_chars_per_section = max_chars_per_section

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

        # Score pages by financial keyword presence
        relevant_page_nums: Set[int] = set()
        for page in non_empty_pages:
            lower_text = page.text.lower()
            if any(kw in lower_text for kw in FINANCIAL_STATEMENT_KEYWORDS):
                relevant_page_nums.add(page.page_number)
                # Include adjacent pages for multi-page tables/notes
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

    def extract_section(self, section_pages: List[ParsedPage]) -> FinancialExtractionReport:
        """Call Gemini to extract structured financial data from a single section.

        Args:
            section_pages: Batch of pages to extract from.

        Returns:
            FinancialExtractionReport for this section.
        """
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

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_config,
            )
        except Exception as api_err:
            raise ExtractionError(f"Gemini API request failed: {api_err}") from api_err

        # Parse output safely
        raw_text = getattr(response, "text", "") or ""
        try:
            parsed_dict = clean_and_parse_llm_json(raw_text)
            return FinancialExtractionReport.model_validate(parsed_dict)
        except Exception as parse_err:
            raise ExtractionError(
                f"Failed to parse or validate structured model output: {parse_err}"
            ) from parse_err

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

        for idx, section in enumerate(sections, start=1):
            page_numbers = [p.page_number for p in section]
            try:
                section_report = self.extract_section(section)
                if not company_name and section_report.company_name:
                    company_name = section_report.company_name

                for incoming_annual in section_report.annual_reports:
                    year = incoming_annual.year
                    if year not in annual_map:
                        annual_map[year] = incoming_annual
                    else:
                        self._merge_annual_data(annual_map[year], incoming_annual)

            except Exception as sec_err:
                msg = f"Section {idx} (pages {page_numbers}) extraction note: {sec_err}"
                logger.warning(msg)
                extraction_notes.append(msg)

        # Sort years chronologically and sanitize
        sorted_annuals = [annual_map[y] for y in sorted(annual_map.keys())]
        for ann in sorted_annuals:
            sanitize_and_flag_annual_data(ann)

        consolidated_report = FinancialExtractionReport(
            company_name=company_name,
            annual_reports=sorted_annuals,
            extraction_notes=extraction_notes,
        )

        # Run validation and append findings to notes
        val_summary = validate_financial_report(consolidated_report)
        for issue in val_summary.issues:
            if issue.severity in ("WARNING", "ERROR"):
                consolidated_report.extraction_notes.append(
                    f"[{issue.severity}] {issue.field} (Year {issue.year}): {issue.message}"
                )

        return consolidated_report
