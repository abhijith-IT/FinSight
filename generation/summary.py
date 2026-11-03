"""AI Executive Summary generation module for FinSight.

Combines verified extracted financial statements, calculated ratios, and
retrieved qualitative report passages into a concise, grounded executive summary
with [Page X] source citations, strengths/weaknesses analysis, and incomplete extraction handling.
"""

from __future__ import annotations

import datetime
import hashlib
import random
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

import config
from analytics.ratios import FinancialRatios
from extraction.financial_schema import (
    AnnualFinancialData,
    FinancialExtractionReport,
    FinancialValue,
)

if TYPE_CHECKING:
    from rag.pipeline import RAGPipeline


SUMMARY_SYSTEM_INSTRUCTION = """You are FinSight's Chief Financial Analyst preparing an authoritative, executive-ready summary of a corporate financial report.

MANDATORY GROUNDING & ANALYTICAL RULES:
1. STRICT GROUNDING: Rely ONLY on the provided verified financial figures, calculated ratios, and report passages. NEVER invent numbers, assume facts not present, or extrapolate beyond stated data.
2. CITATIONS: Every factual statement, corporate initiative, or risk factor derived from the report MUST cite the source page using the format [Page X] (e.g., [Page 4], [Page 12]) corresponding to the page where the evidence was found.
3. ZERO VS UNAVAILABLE: Clearly distinguish unavailable metrics from zero values. Never report unavailable/unreported figures as zero or 0.0, and never treat zero as unavailable. If a metric was not reported (e.g., COGS or Total Debt is unavailable), explicitly state that it was not reported.
4. CALCULATED RATIOS VS REPORTED DATA: Distinguish calculated analytical metrics (e.g. Gross Margin %, Net Profit Margin %, ROE, Debt-to-Equity, YoY Revenue Growth) as financial calculations grounded in the reported numbers.
5. HIGHLIGHT YEAR-OVER-YEAR CHANGES: Highlight significant year-over-year growth, contraction, margin expansion, or margin compression across consecutive fiscal years.
6. STRENGTHS, WEAKNESSES & RISKS: Objectively identify financial strengths (e.g., expanding margins, positive operating cash flow, robust equity base), weaknesses (e.g., rising debt, shrinking margins, negative cash flow), and potential risks cited in the report.
7. INCOMPLETE EXTRACTION NOTICE: If an extraction limitation notice is present indicating partial extraction due to model availability, explicitly state this limitation in the Executive Overview.
8. STRUCTURED BRIEFING FORMAT:
   - ## 🏢 Executive Overview: Company name, reporting years, business scope, headline performance, and any extraction limitations.
   - ## 📈 Revenue & Profitability Trends: Multi-year revenue trajectory, net profit progression, and YoY growth highlights.
   - ## ⚖️ Financial Health & Capital Structure: Balance sheet scale, equity base, debt-to-equity leverage, and cash flow generation.
   - ## 💪 Financial Strengths & Core Drivers: Concrete operational strengths supported by the numbers and [Page X] citations.
   - ## ⚠️ Weaknesses, Strategic Risks & Constraints: Concrete weaknesses, missing data gaps, and principal risk factors identified in the text with [Page X] citations.
9. CONCISE & AUTHORITATIVE: Keep the briefing polished, objective, and structured for C-suite decision-makers.
"""


class ExecutiveSummary(BaseModel):
    """Structured representation of an AI-generated executive summary."""

    model_config = ConfigDict(frozen=True)

    company_name: Optional[str] = Field(
        default=None,
        description="Company name or reporting entity identified from report.",
    )
    reporting_period: Optional[str] = Field(
        default=None,
        description="Reporting fiscal year(s) or period covered.",
    )
    summary_text: str = Field(
        ...,
        description="Markdown formatted executive summary text.",
    )
    cited_pages: List[int] = Field(
        default_factory=list,
        description="Sorted list of unique 1-indexed source pages cited in the summary.",
    )
    generated_at: str = Field(
        default="",
        description="Timestamp when the summary was generated.",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Gemini model name used for generation.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if generation encountered an issue.",
    )
    is_incomplete: bool = Field(
        default=False,
        description="Flag indicating if the summary was generated from an incomplete extraction.",
    )


def format_metric_value(
    fv: Optional[FinancialValue],
    currency: Optional[str] = None,
    unit: Optional[str] = None,
) -> str:
    """Format a FinancialValue, clearly distinguishing unavailable from zero."""
    if fv is None or getattr(fv, "is_unavailable", False) or getattr(fv, "value", None) is None:
        return "Unavailable (Not Reported)"
    val = fv.value
    curr_str = f"{currency} " if currency else ""
    unit_str = f" {unit}" if unit else ""
    if val == 0.0:
        return f"{curr_str}0.00{unit_str} (Zero reported)"
    if abs(val) >= 1_000:
        val_str = f"{val:,.2f}"
    else:
        val_str = f"{val:.2f}"
    return f"{curr_str}{val_str}{unit_str}"


class ExecutiveSummaryGenerator:
    """Generates concise, report-grounded AI executive summaries with retry, fallback, and caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
        max_retries: int = 3,
        base_delay_seconds: float = 0.1,
    ) -> None:
        """Initialize the ExecutiveSummaryGenerator.

        Args:
            api_key: Gemini API key. Defaults to config.GEMINI_API_KEY.
            model_name: Primary Gemini model name. Defaults to config.GEMINI_MODEL.
            fallback_model_name: Fallback model name. Defaults to config.GEMINI_FALLBACK_MODEL.
            client: Optional pre-configured genai.Client instance.
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
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self._cache: Dict[str, ExecutiveSummary] = {}

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

    @staticmethod
    def _compute_cache_key(
        report: FinancialExtractionReport,
        ratios: List[FinancialRatios],
        model_name: str,
    ) -> str:
        """Compute unique deterministic key for report and ratios to enable memory caching."""
        parts = [
            str(report.company_name or ""),
            str(getattr(report, "is_incomplete", False)),
            model_name,
        ]
        for ann in report.annual_reports:
            parts.append(f"{ann.year}_{ann.currency}_{ann.unit}")
            for stmt in [ann.income_statement, ann.balance_sheet, ann.cash_flow_statement]:
                for field in type(stmt).model_fields.keys():
                    fv = getattr(stmt, field, None)
                    val = getattr(fv, "value", None)
                    unavail = getattr(fv, "is_unavailable", False)
                    parts.append(f"{field}:{val}:{unavail}")
        for r in ratios:
            parts.append(f"{r.year}_{r.gross_margin_pct}_{r.net_profit_margin_pct}_{r.roe}_{r.debt_to_equity}_{r.revenue_growth_yoy_pct}")
        return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()

    def generate_summary(
        self,
        report: FinancialExtractionReport,
        ratios: List[FinancialRatios],
        pipeline: Optional[RAGPipeline] = None,
    ) -> ExecutiveSummary:
        """Generate a grounded executive summary for the financial report.

        Args:
            report: Extracted multi-year financial statements.
            ratios: Calculated financial ratios across reporting years.
            pipeline: Optional RAGPipeline for retrieving qualitative disclosures (risks, MD&A).

        Returns:
            ExecutiveSummary object with formatted markdown text and verified citations.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not report or not report.annual_reports:
            return ExecutiveSummary(
                summary_text="No extracted financial statement data available to generate an executive summary.",
                generated_at=now_str,
                error="No annual reports data provided.",
            )

        # Check internal memory cache to avoid repeated API calls
        cache_key = self._compute_cache_key(report, ratios, self.model_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        comp_name = report.company_name or "Reporting Entity"
        years = [a.year for a in report.annual_reports]
        period_str = f"FY {min(years)} – FY {max(years)}" if len(years) > 1 else f"FY {years[0]}"
        is_incomplete_report = getattr(report, "is_incomplete", False)

        # 1. Format verified financial statements context
        statements_context, statement_pages = self._build_statements_context(report)

        # 2. Format calculated financial ratios context
        ratios_context = self._build_ratios_context(ratios)

        # 3. Retrieve qualitative disclosures (business highlights, risks, outlook) from RAG
        qualitative_context, qualitative_pages = self._retrieve_qualitative_context(pipeline)

        all_valid_pages = sorted(statement_pages | qualitative_pages)

        # 4. Construct prompt with explicit analytical sections and limitation notices
        limitation_block = ""
        if is_incomplete_report:
            limitation_block = (
                "\n### IMPORTANT LIMITATION NOTICE:\n"
                "Extraction was partially interrupted due to model availability. "
                "Explicitly communicate this limitation in the Executive Overview.\n"
            )

        prompt = (
            f"# Target Entity: {comp_name}\n"
            f"# Reporting Period: {period_str}\n"
            f"{limitation_block}\n"
            f"### Verified Financial Statement Data:\n"
            f"{statements_context}\n\n"
            f"### Calculated Financial Ratios (Analytical Calculations):\n"
            f"{ratios_context}\n\n"
            f"### Qualitative Report Passages & Disclosures:\n"
            f"{qualitative_context}\n\n"
            f"Generate a comprehensive, executive-ready summary following the mandatory briefing structure. "
            f"Highlight significant year-over-year changes, concrete financial strengths, weaknesses, and key risks. "
            f"Ensure every factual observation cites the source page like [Page X]."
        )

        generate_config = types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
            temperature=0.1,
        )

        # 5. Call Gemini with retry and fallback
        raw_text, model_used, api_error = self._generate_with_retry_and_fallback(prompt, generate_config)

        if api_error or not raw_text.strip():
            # Graceful fallback: construct an automated fact-based summary without crashing
            err_msg = api_error or "Empty response returned from model"
            fallback_text = self._build_fallback_summary(comp_name, period_str, report, ratios, err_msg)
            summary = ExecutiveSummary(
                company_name=comp_name,
                reporting_period=period_str,
                summary_text=fallback_text,
                cited_pages=all_valid_pages,
                generated_at=now_str,
                model_used=model_used or self.model_name,
                error=err_msg,
                is_incomplete=is_incomplete_report,
            )
            self._cache[cache_key] = summary
            return summary

        # 6. Extract and reconcile [Page X] citations against verified pages
        cited_matches = re.findall(r"\[Page\s+(\d+)\]", raw_text, re.IGNORECASE)
        valid_set = set(all_valid_pages)
        cited_pages = sorted({int(p) for p in cited_matches if int(p) in valid_set})

        summary = ExecutiveSummary(
            company_name=comp_name,
            reporting_period=period_str,
            summary_text=raw_text.strip(),
            cited_pages=cited_pages,
            generated_at=now_str,
            model_used=model_used or self.model_name,
            is_incomplete=is_incomplete_report,
        )
        self._cache[cache_key] = summary
        return summary

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
                        delay = self.base_delay_seconds * (2 ** attempt) + random.uniform(0.05, 0.2)
                        time.sleep(delay)
                        continue

            # Retries exhausted for current model; continue to fallback model
            continue

        return "", None, str(last_error) if last_error else "API request failed"

    def _build_statements_context(
        self,
        report: FinancialExtractionReport,
    ) -> tuple[str, Set[int]]:
        """Format annual reports into clean structured text and record source pages."""
        blocks: List[str] = []
        source_pages: Set[int] = set()

        is_incomplete = getattr(report, "is_incomplete", False)
        if is_incomplete:
            blocks.append(
                "### EXTRACTION LIMITATION NOTICE:\n"
                "Financial extraction was partially interrupted due to model availability. "
                "Clearly note in the Executive Overview that some statement sections could not be extracted."
            )

        for ann in report.annual_reports:
            curr = ann.currency or "Reported Currency"
            unit = f" {ann.unit}" if ann.unit else ""
            lines = [f"--- Fiscal Year {ann.year} ({curr}{unit}) ---"]

            # Income Statement
            inc = ann.income_statement
            rev_val = inc.revenue.value if inc.revenue and not inc.revenue.is_unavailable and inc.revenue.value is not None else "Unavailable (Not Reported)"
            cogs_val = inc.cost_of_goods_sold.value if inc.cost_of_goods_sold and not inc.cost_of_goods_sold.is_unavailable and inc.cost_of_goods_sold.value is not None else "Unavailable (Not Reported)"
            op_val = inc.operating_profit.value if inc.operating_profit and not inc.operating_profit.is_unavailable and inc.operating_profit.value is not None else "Unavailable (Not Reported)"
            np_val = inc.net_profit.value if inc.net_profit and not inc.net_profit.is_unavailable and inc.net_profit.value is not None else "Unavailable (Not Reported)"
            lines.append(f"- Revenue: {rev_val} (Source Page: {inc.revenue.source_page if inc.revenue and inc.revenue.source_page else 'N/A'})")
            lines.append(f"- Cost of Goods Sold (COGS): {cogs_val} (Source Page: {inc.cost_of_goods_sold.source_page if inc.cost_of_goods_sold and inc.cost_of_goods_sold.source_page else 'N/A'})")
            lines.append(f"- Operating Profit (EBIT): {op_val} (Source Page: {inc.operating_profit.source_page if inc.operating_profit and inc.operating_profit.source_page else 'N/A'})")
            lines.append(f"- Net Profit (Net Income): {np_val} (Source Page: {inc.net_profit.source_page if inc.net_profit and inc.net_profit.source_page else 'N/A'})")

            # Balance Sheet
            bs = ann.balance_sheet
            ta_val = bs.total_assets.value if bs.total_assets and not bs.total_assets.is_unavailable and bs.total_assets.value is not None else "Unavailable (Not Reported)"
            tl_val = bs.total_liabilities.value if bs.total_liabilities and not bs.total_liabilities.is_unavailable and bs.total_liabilities.value is not None else "Unavailable (Not Reported)"
            debt_val = bs.total_debt.value if bs.total_debt and not bs.total_debt.is_unavailable and bs.total_debt.value is not None else "Unavailable (Not Reported)"
            eq_val = bs.equity.value if bs.equity and not bs.equity.is_unavailable and bs.equity.value is not None else "Unavailable (Not Reported)"
            lines.append(f"- Total Assets: {ta_val} (Source Page: {bs.total_assets.source_page if bs.total_assets and bs.total_assets.source_page else 'N/A'})")
            lines.append(f"- Total Liabilities: {tl_val} (Source Page: {bs.total_liabilities.source_page if bs.total_liabilities and bs.total_liabilities.source_page else 'N/A'})")
            lines.append(f"- Total Debt: {debt_val} (Source Page: {bs.total_debt.source_page if bs.total_debt and bs.total_debt.source_page else 'N/A'})")
            lines.append(f"- Shareholders' Equity: {eq_val} (Source Page: {bs.equity.source_page if bs.equity and bs.equity.source_page else 'N/A'})")

            # Cash Flow
            cf = ann.cash_flow_statement
            ocf_val = cf.operating_cash_flow.value if cf.operating_cash_flow and not cf.operating_cash_flow.is_unavailable and cf.operating_cash_flow.value is not None else "Unavailable (Not Reported)"
            capex_val = cf.capital_expenditure.value if cf.capital_expenditure and not cf.capital_expenditure.is_unavailable and cf.capital_expenditure.value is not None else "Unavailable (Not Reported)"
            lines.append(f"- Operating Cash Flow: {ocf_val} (Source Page: {cf.operating_cash_flow.source_page if cf.operating_cash_flow and cf.operating_cash_flow.source_page else 'N/A'})")
            lines.append(f"- Capital Expenditure (CapEx): {capex_val} (Source Page: {cf.capital_expenditure.source_page if cf.capital_expenditure and cf.capital_expenditure.source_page else 'N/A'})")

            if ann.missing_fields:
                lines.append(f"- Explicitly Unreported / Missing Metrics: {', '.join(ann.missing_fields)}")

            source_pages.update(ann.source_pages)
            for stmt in [inc, bs, cf]:
                for field in type(stmt).model_fields.keys():
                    fv = getattr(stmt, field, None)
                    if fv and getattr(fv, "source_page", None):
                        source_pages.add(fv.source_page)

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks), source_pages

    def _build_ratios_context(self, ratios: List[FinancialRatios]) -> str:
        """Format calculated financial ratios into structured text."""
        if not ratios:
            return "No calculated ratios available."

        lines: List[str] = []
        for r in ratios:
            gm = f"{r.gross_margin_pct:.2f}%" if r.gross_margin_pct is not None else "Unavailable (COGS not reported)"
            npm = f"{r.net_profit_margin_pct:.2f}%" if r.net_profit_margin_pct is not None else "Unavailable"
            roe = f"{r.roe:.2f}%" if r.roe is not None else "Unavailable"
            de = f"{r.debt_to_equity:.2f}x" if r.debt_to_equity is not None else "Unavailable (Total Debt not reported)"
            yoy = f"{r.revenue_growth_yoy_pct:+.2f}%" if r.revenue_growth_yoy_pct is not None else "Unavailable (Requires prior consecutive year)"

            lines.append(f"--- FY {r.year} Calculated Financial Ratios ---")
            lines.append(f"- Gross Margin: {gm}")
            lines.append(f"- Net Profit Margin: {npm}")
            lines.append(f"- Return on Equity (ROE): {roe}")
            lines.append(f"- Debt-to-Equity: {de}")
            lines.append(f"- YoY Revenue Growth: {yoy}")

        return "\n".join(lines)

    def _retrieve_qualitative_context(
        self,
        pipeline: Optional[RAGPipeline],
    ) -> tuple[str, Set[int]]:
        """Retrieve qualitative sections (risks, business overview, outlook) via RAG."""
        if not pipeline or pipeline.chunk_count == 0:
            return "No qualitative document passages indexed.", set()

        search_queries = [
            "executive summary business overview financial performance highlights",
            "principal risk factors uncertainties supply chain competition",
            "strategic outlook future priorities market trends",
        ]

        seen_chunk_ids: Set[str] = set()
        qualitative_blocks: List[str] = []
        source_pages: Set[int] = set()

        for q in search_queries:
            results = pipeline.search(q, top_k=2, min_score=0.1)
            for res in results:
                if res.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(res.chunk_id)
                    source_pages.add(res.page_number)
                    qualitative_blocks.append(
                        f"--- [Page {res.page_number}] ---\n{res.text}"
                    )

        if not qualitative_blocks:
            return "No matching qualitative report passages found.", set()

        return "\n\n".join(qualitative_blocks), source_pages

    def _build_fallback_summary(
        self,
        company_name: str,
        period_str: str,
        report: FinancialExtractionReport,
        ratios: List[FinancialRatios],
        error_msg: str,
    ) -> str:
        """Construct a structured fallback summary if the Gemini API call fails."""
        is_incomplete = getattr(report, "is_incomplete", False)
        lines = [
            f"## 🏢 Executive Overview: {company_name} ({period_str})",
            f"> ⚠️ **Notice:** Real-time AI narrative generation encountered an API issue (`{error_msg}`). "
            f"Displaying automated financial performance breakdown directly from verified statement data.",
        ]

        if is_incomplete:
            failed_info = []
            if getattr(report, "failed_sections", None):
                failed_info.append(f"Failed sections: {report.failed_sections}")
            if getattr(report, "unprocessed_sections", None):
                failed_info.append(f"Skipped sections: {report.unprocessed_sections}")
            detail_str = f" ({'; '.join(failed_info)})" if failed_info else ""
            lines.append(
                f"> ⚠️ **Incomplete Extraction Notice:** Underlying financial extraction was interrupted "
                f"due to model availability limits{detail_str}. Observations reflect only successfully parsed sections."
            )

        lines.extend(["", "## 📈 Revenue & Profitability Trends"])
        for ann in report.annual_reports:
            inc = ann.income_statement
            rev = inc.revenue.value if inc.revenue and not inc.revenue.is_unavailable and inc.revenue.value is not None else "Unavailable (Not Reported)"
            np = inc.net_profit.value if inc.net_profit and not inc.net_profit.is_unavailable and inc.net_profit.value is not None else "Unavailable (Not Reported)"
            curr = ann.currency or ""
            unit = f" {ann.unit}" if ann.unit else ""
            lines.append(f"- **FY {ann.year}:** Revenue = {curr} {rev}{unit} | Net Profit = {curr} {np}{unit}")

        # Highlight significant YoY changes
        ratio_by_year = {r.year: r for r in ratios}
        yoy_highlights = []
        for ann in report.annual_reports:
            r = ratio_by_year.get(ann.year)
            if r and r.revenue_growth_yoy_pct is not None:
                trend = "expansion" if r.revenue_growth_yoy_pct > 0 else "contraction"
                yoy_highlights.append(f"- **FY {ann.year} YoY Growth:** Revenue {trend} of {r.revenue_growth_yoy_pct:+.2f}%.")
        if yoy_highlights:
            lines.extend(["", "### 🚀 Significant Year-over-Year Trajectory:"] + yoy_highlights)

        lines.extend(["", "## ⚖️ Financial Health & Calculated Ratios"])
        for r in ratios:
            gm = f"{r.gross_margin_pct:.2f}%" if r.gross_margin_pct is not None else "Unavailable (COGS not reported)"
            npm = f"{r.net_profit_margin_pct:.2f}%" if r.net_profit_margin_pct is not None else "Unavailable"
            roe = f"{r.roe:.2f}%" if r.roe is not None else "Unavailable"
            de = f"{r.debt_to_equity:.2f}x" if r.debt_to_equity is not None else "Unavailable (Total Debt not reported)"
            yoy = f"{r.revenue_growth_yoy_pct:+.1f}%" if r.revenue_growth_yoy_pct is not None else "N/A"
            lines.append(
                f"- **FY {r.year}:** Gross Margin: {gm} | Net Margin: {npm} | ROE: {roe} | Debt-to-Equity: {de} | YoY Growth: {yoy}"
            )

        # Financial Strengths & Weaknesses
        strengths = []
        weaknesses = []
        for ann in report.annual_reports:
            r = ratio_by_year.get(ann.year)
            cf = ann.cash_flow_statement
            inc = ann.income_statement

            # Strengths
            if r and r.revenue_growth_yoy_pct is not None and r.revenue_growth_yoy_pct > 0:
                strengths.append(f"FY {ann.year}: Positive top-line expansion ({r.revenue_growth_yoy_pct:+.1f}% YoY).")
            if r and r.net_profit_margin_pct is not None and r.net_profit_margin_pct > 10.0:
                strengths.append(f"FY {ann.year}: Healthy net profit margin of {r.net_profit_margin_pct:.1f}%.")
            if cf.operating_cash_flow and cf.operating_cash_flow.value and cf.operating_cash_flow.value > 0:
                strengths.append(f"FY {ann.year}: Robust operating cash flow generation ({ann.currency or ''} {cf.operating_cash_flow.value:,.1f}).")
            if r and r.debt_to_equity is not None and r.debt_to_equity < 1.0:
                strengths.append(f"FY {ann.year}: Prudent capital structure with Debt-to-Equity of {r.debt_to_equity:.2f}x.")

            # Weaknesses / Risks
            if r and r.revenue_growth_yoy_pct is not None and r.revenue_growth_yoy_pct < 0:
                weaknesses.append(f"FY {ann.year}: Top-line revenue contraction of {r.revenue_growth_yoy_pct:+.1f}% YoY.")
            if inc.net_profit and inc.net_profit.value is not None and inc.net_profit.value < 0:
                weaknesses.append(f"FY {ann.year}: Negative net earnings (net loss incurred).")
            if cf.operating_cash_flow and cf.operating_cash_flow.value is not None and cf.operating_cash_flow.value < 0:
                weaknesses.append(f"FY {ann.year}: Operating cash outflow indicates operational cash burn.")
            if r and r.debt_to_equity is not None and r.debt_to_equity > 2.0:
                weaknesses.append(f"FY {ann.year}: Elevated leverage with Debt-to-Equity exceeding 2.0x ({r.debt_to_equity:.2f}x).")
            if ann.missing_fields:
                weaknesses.append(f"FY {ann.year}: Reporting gaps — standard metrics not disclosed: {', '.join(ann.missing_fields)}.")

        lines.extend(["", "## 💪 Financial Strengths & Core Drivers"])
        if strengths:
            lines.extend([f"- {s}" for s in strengths[:5]])
        else:
            lines.append("- Statement data reflects steady performance without outsized margin or growth outperformance.")

        lines.extend(["", "## ⚠️ Weaknesses, Strategic Risks & Constraints"])
        if weaknesses:
            lines.extend([f"- {w}" for w in weaknesses[:5]])
        else:
            lines.append("- No acute leverage or liquidity deficits identified in extracted figures.")

        all_pages = sorted({p for a in report.annual_reports for p in a.source_pages})
        lines.extend([
            "",
            "## 🔍 Data Provenance & Source Pages",
            f"- Extracted data derived from report pages: {', '.join(f'[Page {p}]' for p in all_pages) if all_pages else 'Not recorded'}",
        ])

        return "\n".join(lines)
