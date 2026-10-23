"""FinSight Financial Statement Intelligence Dashboard.

Streamlit application for uploading financial PDF reports, extracting structured
multi-year statements via Gemini, computing financial ratios, and rendering
interactive KPI cards and Plotly trend visualizations.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from analytics.ratios import (
    FinancialRatios,
    calculate_report_ratios,
)
from extraction.extractor import ExtractionError, FinancialExtractor
from extraction.financial_schema import (
    AnnualFinancialData,
    FinancialExtractionReport,
    FinancialValue,
)
from extraction.validation import sanitize_and_flag_annual_data
from ingestion.pdf_parser import (
    EmptyPDFError,
    InvalidPDFError,
    ParsedPDF,
    extract_text_from_pdf,
)
from generation.summary import ExecutiveSummary, ExecutiveSummaryGenerator
from rag.pipeline import RAGPipeline
from rag.qa import QAResult, ReportQAEngine

# Set page configuration
st.set_page_config(
    page_title="FinSight - Financial Statement Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom UI styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-unavailable {
        background-color: #F1F5F9;
        color: #64748B;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-style: italic;
    }
    .source-tag {
        font-size: 0.75rem;
        color: #0284C7;
        background: #E0F2FE;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


REPORT_STATE_KEYS: List[str] = [
    "parsed_pdf",
    "extraction_report",
    "ratios",
    "uploaded_filename",
    "uploaded_file_hash",
    "rag_pipeline",
    "qa_messages",
    "executive_summary",
]


def compute_pdf_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of PDF bytes to uniquely identify document content."""
    return hashlib.sha256(file_bytes).hexdigest()


def extract_file_bytes(uploaded_file: Any) -> bytes:
    """Safely extract bytes from an uploaded file or file-like object."""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        pos = uploaded_file.tell() if hasattr(uploaded_file, "tell") else None
        data = uploaded_file.read()
        if pos is not None and hasattr(uploaded_file, "seek"):
            uploaded_file.seek(pos)
        return data
    raise ValueError("Unsupported uploaded_file type: must support getvalue() or read()")


def clear_report_state() -> None:
    """Clear all report-specific session state to prevent stale data reuse."""
    for key in REPORT_STATE_KEYS:
        st.session_state.pop(key, None)


def format_currency_value(
    value: Optional[float],
    currency: Optional[str] = None,
    unit: Optional[str] = None,
) -> str:
    """Format numeric financial figure with currency and unit or indicate unavailable."""
    if value is None:
        return "Unavailable"

    curr_str = f"{currency} " if currency else ""
    unit_str = f" {unit}" if unit else ""

    # Format large or standard numbers nicely
    if abs(value) >= 1_000:
        val_str = f"{value:,.2f}"
    else:
        val_str = f"{value:.2f}"

    return f"{curr_str}{val_str}{unit_str}"


def format_pct_value(value: Optional[float]) -> str:
    """Format percentage value or indicate unavailable."""
    if value is None:
        return "Unavailable"
    return f"{value:+.2f}%" if value != 0 else "0.00%"


def format_ratio_value(value: Optional[float]) -> str:
    """Format decimal ratio or indicate unavailable."""
    if value is None:
        return "Unavailable"
    return f"{value:.2f}x"


def safe_metric_value(statement: Any, field_name: str) -> Optional[float]:
    """Safely extract metric value from statement object, handling missing attributes or None."""
    if statement is None:
        return None
    metric = getattr(statement, field_name, None)
    if metric is None:
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    return getattr(metric, "value", None)


def safe_metric_source_page(statement: Any, field_name: str) -> Optional[int]:
    """Safely extract source page from statement object."""
    if statement is None:
        return None
    metric = getattr(statement, field_name, None)
    if metric is None:
        return None
    return getattr(metric, "source_page", None)


def safe_metric_obj(statement: Any, field_name: str) -> FinancialValue:
    """Safely get or create FinancialValue object for a metric field."""
    if statement is not None:
        metric = getattr(statement, field_name, None)
        if isinstance(metric, FinancialValue):
            return metric
        if metric is not None and hasattr(metric, "value"):
            return FinancialValue(
                value=getattr(metric, "value", None),
                source_page=getattr(metric, "source_page", None),
                is_unavailable=getattr(metric, "is_unavailable", False),
            )
    return FinancialValue(value=None, is_unavailable=True)


def render_sidebar() -> Tuple[Optional[str], Optional[str]]:
    """Render sidebar branding, API key configuration, and document metadata."""
    with st.sidebar:
        st.title("📈 FinSight")
        st.caption("AI-Powered Financial Statement Intelligence")
        st.markdown("---")

        st.subheader("🔑 API Configuration")
        default_api_key = getattr(config, "GEMINI_API_KEY", "") or ""
        configured_model = getattr(config, "GEMINI_MODEL", "gemini-3.1-flash-lite")

        api_key_input = st.text_input(
            "Gemini API Key",
            value=default_api_key,
            type="password",
            help="Configured in .env or config.py. You can also override it here.",
        )

        model_input = st.selectbox(
            "Gemini Model",
            options=["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=0 if configured_model == "gemini-3.1-flash-lite" else 1,
        )

        st.markdown("---")

        # Document summary if available in session state
        if "parsed_pdf" in st.session_state and st.session_state.parsed_pdf:
            parsed: ParsedPDF = st.session_state.parsed_pdf
            st.subheader("📄 Document Details")
            st.write(f"**File:** {st.session_state.get('uploaded_filename', 'Document')}")
            st.write(f"**Total Pages:** {parsed.total_pages}")
            st.write(f"**Extracted Chars:** {parsed.total_characters:,}")

            if "extraction_report" in st.session_state and st.session_state.extraction_report:
                report: FinancialExtractionReport = st.session_state.extraction_report
                if report.company_name:
                    st.write(f"**Company:** {report.company_name}")
                st.write(f"**Years Extracted:** {len(report.annual_reports)}")

            if st.button("🔄 Clear / Reset", use_container_width=True):
                clear_report_state()
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<small style='color: #94A3B8;'>FinSight Dashboard v1.0</small>",
            unsafe_allow_html=True,
        )

    return api_key_input, model_input


def run_pipeline(
    uploaded_file: Any,
    api_key: Optional[str],
    model_name: str,
    file_bytes: Optional[bytes] = None,
    file_hash: Optional[str] = None,
) -> bool:
    """Execute PDF parsing, LLM financial extraction, and ratio calculation."""
    if not api_key:
        st.error("⚠️ Gemini API Key is required to extract financial statements. Please provide it in the sidebar.")
        return False

    if file_bytes is None:
        file_bytes = extract_file_bytes(uploaded_file)
    if file_hash is None:
        file_hash = compute_pdf_hash(file_bytes)

    filename = getattr(uploaded_file, "name", "document.pdf")

    with st.status("Analyzing Financial Report...", expanded=True) as status:
        # Step 1: PDF Ingestion & Text Parsing
        status.update(label="1/5: Parsing PDF pages with pdfplumber...")
        try:
            pdf_stream = io.BytesIO(file_bytes)
            setattr(pdf_stream, "name", filename)
            parsed_pdf = extract_text_from_pdf(pdf_stream)
        except EmptyPDFError as e:
            st.error(f"Cannot process PDF: {e}")
            status.update(label="Extraction failed: Empty PDF", state="error")
            clear_report_state()
            return False
        except InvalidPDFError as e:
            st.error(f"Invalid or corrupted PDF file: {e}")
            status.update(label="Extraction failed: Corrupted PDF", state="error")
            clear_report_state()
            return False
        except Exception as e:
            st.error(f"Unexpected error while parsing PDF: {e}")
            status.update(label="Extraction failed", state="error")
            clear_report_state()
            return False

        # Step 2: Gemini Structured Statement Extraction
        status.update(label="2/5: Extracting structured financial statements with Gemini...")
        try:
            extractor = FinancialExtractor(
                api_key=api_key,
                model_name=model_name,
                max_pages_per_section=4,
            )
            extraction_report = extractor.extract(parsed_pdf)
        except ExtractionError as e:
            st.error(f"Financial extraction error: {e}")
            status.update(label="Extraction failed: LLM Error", state="error")
            clear_report_state()
            return False
        except Exception as e:
            st.error(f"Failed to extract financial data: {e}")
            status.update(label="Extraction failed", state="error")
            clear_report_state()
            return False

        # Step 3: Pure Python Financial Ratio Engine
        status.update(label="3/5: Computing financial ratios and verifying consistency...")
        try:
            ratios_list = calculate_report_ratios(extraction_report)
        except Exception as e:
            st.error(f"Failed to calculate ratios: {e}")
            status.update(label="Ratio calculation failed", state="error")
            clear_report_state()
            return False

        # Step 4: Index Document in RAG Pipeline for Q&A
        status.update(label="4/5: Indexing document passages for report Q&A...")
        rag_pipeline: Optional[RAGPipeline] = None
        try:
            rag_pipeline = RAGPipeline()
            rag_pipeline.ingest(parsed_pdf, doc_id=filename)
        except Exception as e:
            st.warning(f"Note: RAG indexing encountered an issue: {e}")

        # Step 5: Generate AI Executive Summary
        status.update(label="5/5: Generating AI Executive Summary...")
        exec_summary: Optional[ExecutiveSummary] = None
        try:
            summary_gen = ExecutiveSummaryGenerator(api_key=api_key, model_name=model_name)
            exec_summary = summary_gen.generate_summary(
                report=extraction_report,
                ratios=ratios_list,
                pipeline=rag_pipeline,
            )
        except Exception as e:
            st.warning(f"Note: AI Executive Summary generation encountered an issue ({e}). You can generate it from the dashboard.")

        # Commit fresh report state atomically to session state
        st.session_state.parsed_pdf = parsed_pdf
        st.session_state.uploaded_filename = filename
        st.session_state.uploaded_file_hash = file_hash
        st.session_state.extraction_report = extraction_report
        st.session_state.ratios = ratios_list
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.qa_messages = []
        st.session_state.executive_summary = exec_summary

        status.update(label="Analysis complete!", state="complete", expanded=False)

    return True


def render_kpis(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render top-level KPI cards for selected year."""
    if not annual_reports:
        return

    # Let user select the reporting year for KPI cards (defaults to the most recent)
    available_years = [a.year for a in annual_reports]
    col_title, col_sel = st.columns([3, 1])
    with col_title:
        st.subheader("Key Performance Indicators (KPIs)")
    with col_sel:
        selected_year = st.selectbox(
            "Reporting Year",
            options=available_years,
            index=len(available_years) - 1,
            label_visibility="collapsed",
        )

    # Find matching annual report and ratios
    annual = next((a for a in annual_reports if a.year == selected_year), None)
    ratio_obj = next((r for r in ratios_list if r.year == selected_year), None)

    if not annual:
        return

    curr = annual.currency or ""
    unit = annual.unit or ""

    rev_val = safe_metric_value(annual.income_statement, "revenue")
    np_val = safe_metric_value(annual.income_statement, "net_profit")
    growth_val = ratio_obj.revenue_growth_yoy_pct if ratio_obj else None
    net_margin_val = ratio_obj.net_profit_margin_pct if ratio_obj else None
    roe_val = ratio_obj.roe if ratio_obj else None
    de_val = ratio_obj.debt_to_equity if ratio_obj else None
    gross_margin_val = ratio_obj.gross_margin_pct if ratio_obj else None

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

    with kpi1:
        rev_display = format_currency_value(rev_val, curr, unit)
        growth_display = f"{growth_val:+.1f}% YoY" if growth_val is not None else None
        st.metric(
            label="Total Revenue",
            value=rev_display,
            delta=growth_display,
            help="Reported revenue or net sales for the fiscal year.",
        )
        if rev_val is None:
            st.caption("⚠️ Revenue not reported.")

    with kpi2:
        np_display = format_currency_value(np_val, curr, unit)
        st.metric(
            label="Net Profit",
            value=np_display,
            help="Net income / earnings after tax.",
        )
        if np_val is None:
            st.caption("⚠️ Net Profit not reported.")

    with kpi3:
        gross_margin_display = f"{gross_margin_val:.2f}%" if gross_margin_val is not None else "Unavailable"
        st.metric(
            label="Gross Margin",
            value=gross_margin_display,
            help="(Revenue - COGS) / Revenue × 100",
        )
        if gross_margin_val is None:
            st.caption("ℹ️ COGS not reported.")

    with kpi4:
        margin_display = f"{net_margin_val:.2f}%" if net_margin_val is not None else "Unavailable"
        st.metric(
            label="Net Profit Margin",
            value=margin_display,
            help="Net Profit / Revenue × 100",
        )
        if net_margin_val is None:
            st.caption("⚠️ Requires Net Profit & Revenue.")

    with kpi5:
        roe_display = f"{roe_val:.2f}%" if roe_val is not None else "Unavailable"
        st.metric(
            label="Return on Equity (ROE)",
            value=roe_display,
            help="Net Profit / Shareholders' Equity × 100",
        )
        if roe_val is None:
            st.caption("⚠️ Requires Net Profit & Equity.")

    with kpi6:
        de_display = f"{de_val:.2f}x" if de_val is not None else "Unavailable"
        st.metric(
            label="Debt-to-Equity",
            value=de_display,
            help="Total Debt / Shareholders' Equity. Note: Total liabilities are not treated as debt.",
        )
        if de_val is None:
            st.caption("ℹ️ Total Debt not reported.")


def render_charts(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render interactive Plotly charts for financial metrics and ratios over time."""
    st.subheader("Financial Performance Trends")

    if not annual_reports:
        st.info("No annual financial data available to display charts.")
        return

    # Build DataFrame across all years
    records = []
    ratio_dict: Dict[int, FinancialRatios] = {r.year: r for r in ratios_list}

    for ann in annual_reports:
        y = ann.year
        r = ratio_dict.get(y)
        records.append({
            "Year": str(y),
            "Revenue": safe_metric_value(ann.income_statement, "revenue"),
            "Cost of Goods Sold": safe_metric_value(ann.income_statement, "cost_of_goods_sold"),
            "Net Profit": safe_metric_value(ann.income_statement, "net_profit"),
            "Operating Profit": safe_metric_value(ann.income_statement, "operating_profit"),
            "Operating Expenses": safe_metric_value(ann.income_statement, "operating_expenses"),
            "Gross Margin %": r.gross_margin_pct if r else None,
            "Net Profit Margin %": r.net_profit_margin_pct if r else None,
            "Revenue Growth YoY %": r.revenue_growth_yoy_pct if r else None,
            "ROE %": r.roe if r else None,
            "Currency": ann.currency or "",
            "Unit": ann.unit or "",
        })

    df = pd.DataFrame(records)

    tab_overview, tab_margins, tab_growth = st.tabs([
        "📊 Revenue & Net Profit",
        "📈 Profit Margins & ROE",
        "🚀 YoY Revenue Growth",
    ])

    # Tab 1: Revenue & Net Profit
    with tab_overview:
        has_rev = df["Revenue"].notna().any()
        has_np = df["Net Profit"].notna().any()

        if not has_rev and not has_np:
            st.info("No revenue or net profit data available across reporting years to plot.")
        else:
            fig = go.Figure()
            if has_rev:
                fig.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Revenue"],
                        name="Revenue",
                        marker_color="#2563EB",
                        text=df["Revenue"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )
            if has_np:
                fig.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Net Profit"],
                        name="Net Profit",
                        marker_color="#10B981",
                        text=df["Net Profit"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )

            unit_label = f" ({df['Unit'].iloc[0]})" if not df.empty and df["Unit"].iloc[0] else ""
            curr_label = f" in {df['Currency'].iloc[0]}" if not df.empty and df["Currency"].iloc[0] else ""

            fig.update_layout(
                barmode="group",
                title=f"Revenue and Net Profit Trend{curr_label}{unit_label}",
                xaxis_title="Reporting Fiscal Year",
                yaxis_title=f"Amount{unit_label}",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Profit Margins & ROE
    with tab_margins:
        has_gm = df["Gross Margin %"].notna().any()
        has_margin = df["Net Profit Margin %"].notna().any()
        has_roe = df["ROE %"].notna().any()

        if not has_gm and not has_margin and not has_roe:
            st.info("Profit Margins and Return on Equity data not available to plot.")
        else:
            fig_margin = go.Figure()
            if has_gm:
                fig_margin.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["Gross Margin %"],
                        name="Gross Margin (%)",
                        mode="lines+markers",
                        line=dict(color="#06B6D4", width=3),
                        marker=dict(size=9),
                    )
                )
            if has_margin:
                fig_margin.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["Net Profit Margin %"],
                        name="Net Profit Margin (%)",
                        mode="lines+markers",
                        line=dict(color="#8B5CF6", width=3),
                        marker=dict(size=9),
                    )
                )
            if has_roe:
                fig_margin.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["ROE %"],
                        name="Return on Equity (%)",
                        mode="lines+markers",
                        line=dict(color="#F59E0B", width=2, dash="dash"),
                        marker=dict(size=8),
                    )
                )

            # Add zero benchmark line
            fig_margin.add_hline(y=0, line_dash="dot", line_color="#94A3B8")

            fig_margin.update_layout(
                title="Profitability Ratios Trend (%)",
                xaxis_title="Reporting Fiscal Year",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_margin, use_container_width=True)

    # Tab 3: YoY Revenue Growth
    with tab_growth:
        has_growth = df["Revenue Growth YoY %"].notna().any()
        if not has_growth:
            st.info(
                "Revenue Growth requires at least two consecutive reporting years with reported revenue. "
                "Insufficient consecutive data available to plot growth."
            )
        else:
            # Filter rows with growth data
            growth_df = df[df["Revenue Growth YoY %"].notna()]
            colors = ["#10B981" if val >= 0 else "#EF4444" for val in growth_df["Revenue Growth YoY %"]]

            fig_growth = go.Figure()
            fig_growth.add_trace(
                go.Bar(
                    x=growth_df["Year"],
                    y=growth_df["Revenue Growth YoY %"],
                    name="YoY Revenue Growth %",
                    marker_color=colors,
                    text=growth_df["Revenue Growth YoY %"].apply(lambda v: f"{v:+.1f}%"),
                    textposition="auto",
                )
            )

            fig_growth.add_hline(y=0, line_dash="solid", line_color="#64748B", line_width=1)

            fig_growth.update_layout(
                title="Year-over-Year (YoY) Revenue Growth Rate (%)",
                xaxis_title="Reporting Fiscal Year",
                yaxis_title="Growth Rate (%)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_growth, use_container_width=True)


def render_financial_tables(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render structured financial statement tables grouped by reporting year with source audit tags."""
    st.subheader("Extracted Financial Statements")

    if not annual_reports:
        st.info("No financial statements extracted.")
        return

    years = [a.year for a in annual_reports]
    ratio_dict: Dict[int, FinancialRatios] = {r.year: r for r in ratios_list}

    tab_summary, tab_is, tab_bs, tab_cf, tab_ratios = st.tabs([
        "📋 Full Overview",
        "💵 Income Statement",
        "⚖️ Balance Sheet",
        "🔄 Cash Flow Statement",
        "📐 Financial Ratios",
    ])

    # Tab 1: Full Overview
    with tab_summary:
        summary_rows = [
            ("Revenue", lambda a: safe_metric_value(a.income_statement, "revenue"), lambda a: safe_metric_source_page(a.income_statement, "revenue")),
            ("Cost of Goods Sold (COGS)", lambda a: safe_metric_value(a.income_statement, "cost_of_goods_sold"), lambda a: safe_metric_source_page(a.income_statement, "cost_of_goods_sold")),
            ("Operating Expenses", lambda a: safe_metric_value(a.income_statement, "operating_expenses"), lambda a: safe_metric_source_page(a.income_statement, "operating_expenses")),
            ("Operating Profit", lambda a: safe_metric_value(a.income_statement, "operating_profit"), lambda a: safe_metric_source_page(a.income_statement, "operating_profit")),
            ("Net Profit", lambda a: safe_metric_value(a.income_statement, "net_profit"), lambda a: safe_metric_source_page(a.income_statement, "net_profit")),
            ("Total Assets", lambda a: safe_metric_value(a.balance_sheet, "total_assets"), lambda a: safe_metric_source_page(a.balance_sheet, "total_assets")),
            ("Total Liabilities", lambda a: safe_metric_value(a.balance_sheet, "total_liabilities"), lambda a: safe_metric_source_page(a.balance_sheet, "total_liabilities")),
            ("Total Debt", lambda a: safe_metric_value(a.balance_sheet, "total_debt"), lambda a: safe_metric_source_page(a.balance_sheet, "total_debt")),
            ("Shareholders' Equity", lambda a: safe_metric_value(a.balance_sheet, "equity"), lambda a: safe_metric_source_page(a.balance_sheet, "equity")),
            ("Operating Cash Flow", lambda a: safe_metric_value(a.cash_flow_statement, "operating_cash_flow"), lambda a: safe_metric_source_page(a.cash_flow_statement, "operating_cash_flow")),
            ("Capital Expenditure (CapEx)", lambda a: safe_metric_value(a.cash_flow_statement, "capital_expenditure"), lambda a: safe_metric_source_page(a.cash_flow_statement, "capital_expenditure")),
        ]

        table_data = []
        for name, getter, page_getter in summary_rows:
            row: Dict[str, str] = {"Metric": name}
            for a in annual_reports:
                val = getter(a)
                pg = page_getter(a)
                val_str = format_currency_value(val, a.currency, a.unit)
                pg_str = f" (p.{pg})" if pg else ""
                row[str(a.year)] = f"{val_str}{pg_str}"
            table_data.append(row)

        st.dataframe(pd.DataFrame(table_data).set_index("Metric"), use_container_width=True)

    # Tab 2: Income Statement
    with tab_is:
        is_items = [
            ("Revenue", lambda a: safe_metric_obj(a.income_statement, "revenue")),
            ("Cost of Goods Sold (COGS)", lambda a: safe_metric_obj(a.income_statement, "cost_of_goods_sold")),
            ("Operating Expenses (OPEX)", lambda a: safe_metric_obj(a.income_statement, "operating_expenses")),
            ("Operating Profit (EBIT)", lambda a: safe_metric_obj(a.income_statement, "operating_profit")),
            ("Net Profit (Net Income)", lambda a: safe_metric_obj(a.income_statement, "net_profit")),
        ]
        is_data = []
        for label, metric_fn in is_items:
            row = {"Income Statement Metric": label}
            for a in annual_reports:
                metric = metric_fn(a)
                row[str(a.year)] = format_currency_value(metric.value, a.currency, a.unit)
                row[f"{a.year} Source Page"] = f"Page {metric.source_page}" if metric.source_page else "N/A"
            is_data.append(row)
        st.dataframe(pd.DataFrame(is_data).set_index("Income Statement Metric"), use_container_width=True)

    # Tab 3: Balance Sheet
    with tab_bs:
        bs_items = [
            ("Total Assets", lambda a: safe_metric_obj(a.balance_sheet, "total_assets")),
            ("Total Liabilities", lambda a: safe_metric_obj(a.balance_sheet, "total_liabilities")),
            ("Total Debt", lambda a: safe_metric_obj(a.balance_sheet, "total_debt")),
            ("Shareholders' Equity", lambda a: safe_metric_obj(a.balance_sheet, "equity")),
        ]
        bs_data = []
        for label, metric_fn in bs_items:
            row = {"Balance Sheet Metric": label}
            for a in annual_reports:
                metric = metric_fn(a)
                row[str(a.year)] = format_currency_value(metric.value, a.currency, a.unit)
                row[f"{a.year} Source Page"] = f"Page {metric.source_page}" if metric.source_page else "N/A"
            bs_data.append(row)
        st.dataframe(pd.DataFrame(bs_data).set_index("Balance Sheet Metric"), use_container_width=True)

    # Tab 4: Cash Flow Statement
    with tab_cf:
        cf_items = [
            ("Operating Cash Flow", lambda a: safe_metric_obj(a.cash_flow_statement, "operating_cash_flow")),
            ("Capital Expenditure (CapEx)", lambda a: safe_metric_obj(a.cash_flow_statement, "capital_expenditure")),
        ]
        cf_data = []
        for label, metric_fn in cf_items:
            row = {"Cash Flow Metric": label}
            for a in annual_reports:
                metric = metric_fn(a)
                row[str(a.year)] = format_currency_value(metric.value, a.currency, a.unit)
                row[f"{a.year} Source Page"] = f"Page {metric.source_page}" if metric.source_page else "N/A"
            cf_data.append(row)
        st.dataframe(pd.DataFrame(cf_data).set_index("Cash Flow Metric"), use_container_width=True)

    # Tab 5: Financial Ratios
    with tab_ratios:
        ratio_rows = [
            ("Gross Margin %", lambda r: format_pct_value(r.gross_margin_pct)),
            ("Net Profit Margin %", lambda r: format_pct_value(r.net_profit_margin_pct)),
            ("Return on Equity (ROE) %", lambda r: format_pct_value(r.roe)),
            ("Debt-to-Equity", lambda r: format_ratio_value(r.debt_to_equity)),
            ("Revenue Growth YoY %", lambda r: format_pct_value(r.revenue_growth_yoy_pct)),
        ]
        ratio_table = []
        for name, formatter in ratio_rows:
            row = {"Financial Ratio": name}
            for y in years:
                r_obj = ratio_dict.get(y)
                row[str(y)] = formatter(r_obj) if r_obj else "Unavailable"
            ratio_table.append(row)
        st.dataframe(pd.DataFrame(ratio_table).set_index("Financial Ratio"), use_container_width=True)


def get_or_create_rag_pipeline(parsed_pdf: Optional[ParsedPDF]) -> Optional[RAGPipeline]:
    """Retrieve existing RAG pipeline from session state or index the parsed PDF."""
    if "rag_pipeline" in st.session_state and st.session_state.rag_pipeline is not None:
        return st.session_state.rag_pipeline
    if parsed_pdf is not None and parsed_pdf.pages:
        pipeline = RAGPipeline()
        pipeline.ingest(parsed_pdf)
        st.session_state.rag_pipeline = pipeline
        return pipeline
    return None


def render_qa_chat(api_key: Optional[str], model_name: str) -> None:
    """Render interactive report-grounded Q&A chat section."""
    st.subheader("💬 Ask Questions About This Report")
    st.caption(
        "Ask questions about financial performance, risks, disclosures, or footnotes. "
        "Answers are strictly grounded in retrieved passages from the uploaded report with source page citations (e.g., [Page 4])."
    )

    # Initialize chat history if not present
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Get or lazily initialize the RAG pipeline from the cached parsed_pdf
    parsed_pdf = st.session_state.get("parsed_pdf")
    pipeline = get_or_create_rag_pipeline(parsed_pdf)

    # Status / Info and clear conversation
    col_info, col_clear = st.columns([4, 1])
    with col_info:
        if pipeline and pipeline.chunk_count > 0:
            st.markdown(
                f"<small style='color: #64748B;'>Indexed <b>{pipeline.chunk_count}</b> passages across <b>{len(pipeline.indexed_pages)}</b> pages ready for Q&A.</small>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No report text is currently indexed for Q&A.")
            return

    with col_clear:
        if st.session_state.qa_messages:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
                st.session_state.qa_messages = []
                st.rerun()

    # Display chat history
    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display source citations expander if available
            sources = msg.get("sources")
            if sources:
                with st.expander(f"📄 Retrieved Sources ({len(sources)} passages)", expanded=False):
                    for idx, src in enumerate(sources):
                        st.markdown(f"**[Page {src['page_number']}]** (Score: `{src['score']:.2f}`):")
                        st.markdown(f"> *{src['text']}*")
                        if idx < len(sources) - 1:
                            st.markdown("---")

    # Chat input
    user_query = st.chat_input("Ask a question about this report (e.g., 'What are the principal risk factors?')...")
    if user_query:
        if not api_key:
            st.error("⚠️ Gemini API key is required to answer questions. Please provide it in the sidebar.")
            return

        # Record user question
        st.session_state.qa_messages.append({"role": "user", "content": user_query})

        with st.spinner("Searching report passages and generating grounded answer..."):
            qa_engine = ReportQAEngine(
                api_key=api_key,
                model_name=model_name,
                top_k=5,
            )
            qa_result = qa_engine.answer_question(user_query, pipeline)

            # Format source summaries
            source_summaries = [
                {
                    "page_number": r.page_number,
                    "score": r.score,
                    "text": r.text[:350] + ("..." if len(r.text) > 350 else ""),
                }
                for r in qa_result.sources
            ]

            st.session_state.qa_messages.append({
                "role": "assistant",
                "content": qa_result.answer,
                "sources": source_summaries,
                "cited_pages": qa_result.cited_pages,
                "has_insufficient_evidence": qa_result.has_insufficient_evidence,
            })

        st.rerun()


def render_executive_summary_section(
    report: FinancialExtractionReport,
    ratios: List[FinancialRatios],
    api_key: Optional[str],
    model_name: str,
) -> None:
    """Render prominent AI Executive Summary briefing with cached generation."""
    st.subheader("📋 AI Executive Summary")

    cached_summary: Optional[ExecutiveSummary] = st.session_state.get("executive_summary")

    # Header control row: Metadata on left, Regenerate / Generate button on right
    col_meta, col_btn = st.columns([4, 1])

    with col_meta:
        if cached_summary and not cached_summary.error:
            cited_str = f" • Sources: {', '.join(f'Page {p}' for p in cached_summary.cited_pages)}" if cached_summary.cited_pages else ""
            st.caption(f"🤖 Generated by **{cached_summary.model_used or model_name}** at {cached_summary.generated_at}{cited_str}")
        else:
            st.caption("AI strategic briefing combining verified statement data, calculated ratios, and reported management disclosures.")

    with col_btn:
        regen_label = "🔄 Regenerate" if cached_summary else "✨ Generate Summary"
        if st.button(regen_label, key="gen_summary_btn", use_container_width=True):
            if not api_key:
                st.error("⚠️ Gemini API key is required to generate the executive summary. Please configure it in the sidebar.")
                return
            with st.spinner("Analyzing multi-year performance and synthesizing executive briefing..."):
                pipeline = get_or_create_rag_pipeline(st.session_state.get("parsed_pdf"))
                generator = ExecutiveSummaryGenerator(api_key=api_key, model_name=model_name)
                st.session_state.executive_summary = generator.generate_summary(report, ratios, pipeline)
            st.rerun()

    # Display cached summary
    cached_summary = st.session_state.get("executive_summary")
    if cached_summary:
        if cached_summary.error:
            st.warning(f"Executive Summary Notice: {cached_summary.error}")
        with st.container(border=True):
            st.markdown(cached_summary.summary_text)
    else:
        st.info("Click **✨ Generate Summary** to synthesize an executive briefing for this report.")


def render_audit_and_validation(report: FinancialExtractionReport) -> None:
    """Render audit notices, missing fields breakdown, and extraction warnings."""
    with st.expander("🔍 Extraction & Audit Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Report Provenance & Metadata**")
            st.write(f"- **Identified Entity:** {report.company_name or 'Not explicitly identified'}")
            st.write(f"- **Reporting Years Found:** {', '.join(str(a.year) for a in report.annual_reports)}")
            for a in report.annual_reports:
                source_pgs = ", ".join(map(str, sorted(a.source_pages))) or "None recorded"
                st.write(f"- **FY {a.year} Source Pages:** {source_pgs}")

        with col2:
            st.markdown("**Unavailable / Missing Metrics by Year**")
            for a in report.annual_reports:
                if a.missing_fields:
                    fields_list = ", ".join(a.missing_fields)
                    st.write(f"- **FY {a.year}:** {fields_list}")
                else:
                    st.write(f"- **FY {a.year}:** All 9 standard statement figures reported.")

        if report.extraction_notes:
            st.markdown("---")
            st.markdown("**Extraction Notes & Accounting Consistency Checks:**")
            for note in report.extraction_notes:
                st.caption(f"• {note}")


def main() -> None:
    """Main Streamlit execution routine."""
    api_key, model_name = render_sidebar()

    st.markdown('<div class="main-header">FinSight Financial Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload an annual report or 10-K PDF to extract financial statements and analyze financial performance.</div>',
        unsafe_allow_html=True,
    )

    # PDF File Uploader
    uploaded_file = st.file_uploader(
        "Choose an annual report PDF file",
        type=["pdf"],
        help="Upload a corporate financial report or 10-K document (PDF).",
    )

    if uploaded_file is None:
        # If user removed the file, clear any lingering report state and prompt upload
        if st.session_state.get("uploaded_file_hash") is not None or "extraction_report" in st.session_state:
            clear_report_state()
            st.rerun()
        st.info("👆 Upload an annual report PDF to begin extracting and analyzing financial statements.")
        return

    # Process upload from actual bytes and content hash
    file_bytes = extract_file_bytes(uploaded_file)
    file_hash = compute_pdf_hash(file_bytes)
    current_hash = st.session_state.get("uploaded_file_hash")

    # If the PDF changed (different content hash), clear all report-specific state immediately
    if current_hash is not None and current_hash != file_hash:
        clear_report_state()
        st.rerun()

    # Rebuild extraction if not yet extracted for this document
    if "extraction_report" not in st.session_state or current_hash != file_hash:
        process_btn = st.button("🚀 Analyze Report", type="primary", use_container_width=True)
        if process_btn:
            success = run_pipeline(
                uploaded_file=uploaded_file,
                api_key=api_key,
                model_name=model_name,
                file_bytes=file_bytes,
                file_hash=file_hash,
            )
            if not success:
                st.stop()
            st.rerun()
        else:
            st.info(f"Ready to analyze **{uploaded_file.name}**. Click **Analyze Report** above to start.")
            st.stop()

    # Display Dashboard Results only if extraction matches current uploaded file hash
    if (
        "extraction_report" in st.session_state
        and st.session_state.extraction_report
        and st.session_state.get("uploaded_file_hash") == file_hash
    ):
        report: FinancialExtractionReport = st.session_state.extraction_report

        # Ensure all annual reports conform to schema (reconcile any older cached session objects)
        for ann in report.annual_reports:
            sanitize_and_flag_annual_data(ann)

        if "ratios" not in st.session_state or not st.session_state.ratios:
            st.session_state.ratios = calculate_report_ratios(report)

        ratios: List[FinancialRatios] = st.session_state.get("ratios", [])

        if not report.annual_reports:
            st.warning(
                "No structured financial statements could be extracted from this document. "
                "The report might be an image-only scan or does not contain standard financial statement tables."
            )
            render_audit_and_validation(report)
            return

        st.markdown("---")

        # Top-level entity banner
        comp_name = report.company_name or "Financial Report Analysis"
        year_span = f"FY {report.annual_reports[0].year} – FY {report.annual_reports[-1].year}"
        curr = report.annual_reports[0].currency or ""
        unit = report.annual_reports[0].unit or ""
        scale_tag = f" | {curr} ({unit})" if curr or unit else ""

        st.markdown(f"### {comp_name} <small style='color: #64748B;'>({year_span}{scale_tag})</small>", unsafe_allow_html=True)

        # AI Executive Summary Briefing
        render_executive_summary_section(report, ratios, api_key, model_name)
        st.markdown("---")

        # 1. KPI Cards
        render_kpis(report.annual_reports, ratios)
        st.markdown("---")

        # 2. Interactive Plotly Charts
        render_charts(report.annual_reports, ratios)
        st.markdown("---")

        # 3. Extracted Statements Data Tables
        render_financial_tables(report.annual_reports, ratios)
        st.markdown("---")

        # 4. Report-Grounded Q&A Chat
        render_qa_chat(api_key, model_name)
        st.markdown("---")

        # 5. Audit & Validation Notices
        render_audit_and_validation(report)


if __name__ == "__main__":
    main()
