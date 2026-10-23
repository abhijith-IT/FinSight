"""AI Executive Summary generation module for FinSight.

Combines verified extracted financial statements, calculated ratios, and
retrieved qualitative report passages into a concise, grounded executive summary
with [Page X] source citations.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Set

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

import config
from analytics.ratios import FinancialRatios
from extraction.financial_schema import (
    AnnualFinancialData,
    FinancialExtractionReport,
)
from rag.pipeline import RAGPipeline


SUMMARY_SYSTEM_INSTRUCTION = """You are FinSight's Chief Financial Analyst preparing an authoritative, executive-ready summary of a corporate financial report.

MANDATORY GROUNDING RULES:
1. STRICT GROUNDING: Rely ONLY on the provided financial figures and report passages. NEVER invent numbers, assume facts not present, or speculate on unmentioned events.
2. CITATIONS: Every operational statement, corporate initiative, or risk factor derived from the report MUST cite the source page using the format [Page X] (e.g., [Page 4], [Page 12]) corresponding to the page heading where the evidence was found.
3. CALCULATED METRICS VS REPORTED STATEMENTS: Clearly distinguish calculated financial ratios (e.g., Gross Margin %, Net Profit Margin %, ROE, Debt-to-Equity, YoY Revenue Growth) as financial calculations rather than direct quotes from the text.
4. MISSING METRICS: If a metric was not reported (e.g., Cost of Goods Sold or Total Debt is unavailable), explicitly note that it was not reported. Do NOT invent or infer missing values.
5. PROFESSIONAL BRIEFING STRUCTURE:
   - ## 🏢 Executive Overview: Entity name, reporting period, headline performance, and core business scope.
   - ## 📈 Revenue & Profitability Trends: Multi-year revenue trajectory, net profit progression, and margin evolution.
   - ## ⚖️ Financial Health & Capital Structure: Balance sheet scale, debt-to-equity leverage, equity base, and cash flow generation.
   - ## ⚠️ Strategic Highlights & Key Risks: Management highlights and principal risk factors explicitly identified in the report with [Page X] citations.
6. CONCISE & AUTHORITATIVE: Keep the briefing concise, polished, and structured for C-suite decision-makers.
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


class ExecutiveSummaryGenerator:
    """Generates concise, report-grounded AI executive summaries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        """Initialize the ExecutiveSummaryGenerator.

        Args:
            api_key: Gemini API key. Defaults to config.GEMINI_API_KEY.
            model_name: Gemini model name. Defaults to config.GEMINI_MODEL.
            client: Optional pre-configured genai.Client instance (useful for testing).
        """
        self.api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
        self.model_name = model_name or getattr(config, "GEMINI_MODEL", "gemini-3.1-flash-lite")
        self._client = client

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

        comp_name = report.company_name or "Reporting Entity"
        years = [a.year for a in report.annual_reports]
        period_str = f"FY {min(years)} – FY {max(years)}" if len(years) > 1 else f"FY {years[0]}"

        # 1. Format verified financial statements context
        statements_context, statement_pages = self._build_statements_context(report)

        # 2. Format calculated financial ratios context
        ratios_context = self._build_ratios_context(ratios)

        # 3. Retrieve qualitative disclosures (business highlights, risks, outlook) from RAG
        qualitative_context, qualitative_pages = self._retrieve_qualitative_context(pipeline)

        all_valid_pages = sorted(statement_pages | qualitative_pages)

        # 4. Construct prompt
        prompt = (
            f"# Target Entity: {comp_name}\n"
            f"# Reporting Period: {period_str}\n\n"
            f"### Verified Financial Statement Data:\n"
            f"{statements_context}\n\n"
            f"### Calculated Financial Ratios (Analytical Calculations):\n"
            f"{ratios_context}\n\n"
            f"### Qualitative Report Passages & Disclosures:\n"
            f"{qualitative_context}\n\n"
            f"Generate a comprehensive, executive-ready summary following the mandatory briefing structure. "
            f"Ensure every qualitative observation or risk cites the source page like [Page X]."
        )

        generate_config = types.GenerateContentConfig(
            system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
            temperature=0.1,
        )

        # 5. Call Gemini
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_config,
            )
            raw_text = getattr(response, "text", "") or ""
            if not raw_text.strip():
                raw_text = "The model returned an empty summary."
        except Exception as api_err:
            # Graceful fallback: construct an automated fact-based summary without crashing
            fallback_text = self._build_fallback_summary(comp_name, period_str, report, ratios, str(api_err))
            return ExecutiveSummary(
                company_name=comp_name,
                reporting_period=period_str,
                summary_text=fallback_text,
                cited_pages=all_valid_pages,
                generated_at=now_str,
                model_used=self.model_name,
                error=str(api_err),
            )

        # 6. Extract and reconcile [Page X] citations
        cited_matches = re.findall(r"\[Page\s+(\d+)\]", raw_text, re.IGNORECASE)
        valid_set = set(all_valid_pages)
        cited_pages = sorted({int(p) for p in cited_matches if int(p) in valid_set})

        return ExecutiveSummary(
            company_name=comp_name,
            reporting_period=period_str,
            summary_text=raw_text.strip(),
            cited_pages=cited_pages,
            generated_at=now_str,
            model_used=self.model_name,
        )

    def _build_statements_context(
        self,
        report: FinancialExtractionReport,
    ) -> tuple[str, Set[int]]:
        """Format annual reports into clean structured text and record source pages."""
        blocks: List[str] = []
        source_pages: Set[int] = set()

        for ann in report.annual_reports:
            curr = ann.currency or "Reported Currency"
            unit = f" {ann.unit}" if ann.unit else ""
            lines = [f"--- Fiscal Year {ann.year} ({curr}{unit}) ---"]

            # Income Statement
            inc = ann.income_statement
            rev_val = inc.revenue.value if inc.revenue and not inc.revenue.is_unavailable else "Unavailable"
            cogs_val = inc.cost_of_goods_sold.value if inc.cost_of_goods_sold and not inc.cost_of_goods_sold.is_unavailable else "Unavailable"
            op_val = inc.operating_profit.value if inc.operating_profit and not inc.operating_profit.is_unavailable else "Unavailable"
            np_val = inc.net_profit.value if inc.net_profit and not inc.net_profit.is_unavailable else "Unavailable"
            lines.append(f"- Revenue: {rev_val} (Source Page: {inc.revenue.source_page if inc.revenue else 'N/A'})")
            lines.append(f"- Cost of Goods Sold (COGS): {cogs_val} (Source Page: {inc.cost_of_goods_sold.source_page if inc.cost_of_goods_sold else 'N/A'})")
            lines.append(f"- Operating Profit (EBIT): {op_val} (Source Page: {inc.operating_profit.source_page if inc.operating_profit else 'N/A'})")
            lines.append(f"- Net Profit (Net Income): {np_val} (Source Page: {inc.net_profit.source_page if inc.net_profit else 'N/A'})")

            # Balance Sheet
            bs = ann.balance_sheet
            ta_val = bs.total_assets.value if bs.total_assets and not bs.total_assets.is_unavailable else "Unavailable"
            tl_val = bs.total_liabilities.value if bs.total_liabilities and not bs.total_liabilities.is_unavailable else "Unavailable"
            debt_val = bs.total_debt.value if bs.total_debt and not bs.total_debt.is_unavailable else "Unavailable"
            eq_val = bs.equity.value if bs.equity and not bs.equity.is_unavailable else "Unavailable"
            lines.append(f"- Total Assets: {ta_val} (Source Page: {bs.total_assets.source_page if bs.total_assets else 'N/A'})")
            lines.append(f"- Total Liabilities: {tl_val} (Source Page: {bs.total_liabilities.source_page if bs.total_liabilities else 'N/A'})")
            lines.append(f"- Total Debt: {debt_val} (Source Page: {bs.total_debt.source_page if bs.total_debt else 'N/A'})")
            lines.append(f"- Shareholders' Equity: {eq_val} (Source Page: {bs.equity.source_page if bs.equity else 'N/A'})")

            # Cash Flow
            cf = ann.cash_flow_statement
            ocf_val = cf.operating_cash_flow.value if cf.operating_cash_flow and not cf.operating_cash_flow.is_unavailable else "Unavailable"
            capex_val = cf.capital_expenditure.value if cf.capital_expenditure and not cf.capital_expenditure.is_unavailable else "Unavailable"
            lines.append(f"- Operating Cash Flow: {ocf_val} (Source Page: {cf.operating_cash_flow.source_page if cf.operating_cash_flow else 'N/A'})")
            lines.append(f"- Capital Expenditure (CapEx): {capex_val} (Source Page: {cf.capital_expenditure.source_page if cf.capital_expenditure else 'N/A'})")

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
        lines = [
            f"## 🏢 Executive Overview: {company_name} ({period_str})",
            f"> ⚠️ **Note:** Real-time AI narrative generation encountered an API issue (`{error_msg}`). "
            f"Displaying automated financial performance breakdown directly from verified data.",
            "",
            "## 📈 Revenue & Profitability Trends",
        ]

        for ann in report.annual_reports:
            inc = ann.income_statement
            rev = inc.revenue.value if inc.revenue and inc.revenue.value is not None else "N/A"
            np = inc.net_profit.value if inc.net_profit and inc.net_profit.value is not None else "N/A"
            curr = ann.currency or ""
            unit = f" {ann.unit}" if ann.unit else ""
            lines.append(f"- **FY {ann.year}:** Revenue = {curr} {rev}{unit} | Net Profit = {curr} {np}{unit}")

        lines.extend(["", "## ⚖️ Financial Health & Calculated Ratios"])
        for r in ratios:
            npm = f"{r.net_profit_margin_pct:.2f}%" if r.net_profit_margin_pct is not None else "N/A"
            roe = f"{r.roe:.2f}%" if r.roe is not None else "N/A"
            de = f"{r.debt_to_equity:.2f}x" if r.debt_to_equity is not None else "N/A"
            yoy = f"{r.revenue_growth_yoy_pct:+.1f}%" if r.revenue_growth_yoy_pct is not None else "N/A"
            lines.append(
                f"- **FY {r.year}:** Net Margin: {npm} | ROE: {roe} | Debt-to-Equity: {de} | YoY Revenue Growth: {yoy}"
            )

        lines.extend([
            "",
            "## 🔍 Data Provenance",
            f"- Extracted data derived from report pages: {', '.join(map(str, sorted({p for a in report.annual_reports for p in a.source_pages})))}",
        ])

        return "\n".join(lines)
