"""FinSight Financial Statement Intelligence Dashboard.

Enterprise SaaS Dark Financial Analytics Platform for uploading corporate reports,
extracting multi-year structured statements via Gemini, computing financial ratios,
and rendering institutional-grade KPI cards, Plotly dark visualizations, AI insights,
and grounded RAG financial research.
"""

from __future__ import annotations

import hashlib
import io
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from generation.export import generate_consolidated_csv, generate_executive_pdf
from generation.summary import ExecutiveSummary, ExecutiveSummaryGenerator
try:
    from rag.pipeline import RAGPipeline, create_rag_pipeline
except ImportError:
    from rag import RAGPipeline, create_rag_pipeline
from rag.qa import QAResult, ReportQAEngine

# Set page configuration with wide layout
st.set_page_config(
    page_title="FinSight | Financial Statement Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom UI styling - Premium Dark Financial Dashboard Design System
st.markdown(
    """
    <style>
    /* =========================================================================
       FinSight Dark Financial Dashboard Design System
       Enterprise SaaS Architecture (Deep Charcoal / Slate / Electric Cyan)
       ========================================================================= */

    :root {
        --bg-main: #0B0F19;
        --bg-surface: #111827;
        --bg-surface-elevated: #161F30;
        --bg-surface-highlight: #1E293B;
        --border-subtle: #1F2937;
        --border-medium: #334155;
        --border-accent: rgba(56, 189, 248, 0.35);
        
        --accent-cyan: #38BDF8;
        --accent-blue: #2563EB;
        --accent-cyan-glow: rgba(56, 189, 248, 0.12);
        
        --status-success: #10B981;
        --status-success-bg: rgba(16, 185, 129, 0.12);
        --status-danger: #EF4444;
        --status-danger-bg: rgba(239, 68, 68, 0.12);
        --status-warning: #F59E0B;
        --status-warning-bg: rgba(245, 158, 11, 0.12);
        
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }

    /* Base Body and Streamlit container styling */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar container styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary);
    }

    /* Sidebar brand header */
    .sidebar-brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 18px;
    }
    .sidebar-logo-badge {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.1rem;
        width: 36px;
        height: 36px;
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.3);
    }
    .sidebar-brand-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .sidebar-brand-sub {
        font-size: 0.74rem;
        color: var(--accent-cyan);
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Navigation Radio Items in Sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: transparent;
        padding: 6px 10px;
        border-radius: var(--radius-sm);
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: var(--bg-surface-elevated);
        color: var(--text-primary) !important;
    }

    /* Institutional Header Banner */
    .header-banner-card {
        background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-surface-elevated) 100%);
        border: 1px solid var(--border-subtle);
        border-left: 4px solid var(--accent-cyan);
        border-radius: var(--radius-lg);
        padding: 18px 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .header-entity-name {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .header-meta-row {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 8px;
        font-size: 0.84rem;
        color: var(--text-secondary);
    }
    .header-meta-pill {
        background-color: var(--bg-surface-highlight);
        border: 1px solid var(--border-subtle);
        padding: 2px 10px;
        border-radius: var(--radius-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    /* Status Badges */
    .status-pill-verified {
        background-color: var(--status-success-bg);
        color: var(--status-success);
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-pill-incomplete {
        background-color: var(--status-warning-bg);
        color: var(--status-warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Landing Welcome Page */
    .landing-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
        padding: 20px 0 16px 0;
        margin-left: auto;
        margin-right: auto;
    }
    .landing-brand-tag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: var(--accent-cyan-glow);
        border: 1px solid var(--border-accent);
        color: var(--accent-cyan);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 12px;
        text-align: center;
    }
    .landing-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.03em;
        margin: 0 auto 12px auto;
        text-align: center;
        width: 100%;
        max-width: 860px;
        line-height: 1.25;
    }
    .landing-title span {
        color: var(--accent-cyan);
    }
    .landing-subtitle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
        max-width: 640px;
        margin-left: auto;
        margin-right: auto;
    }
    .landing-subtitle {
        font-size: 1.0rem;
        color: var(--text-secondary);
        width: 100%;
        max-width: 640px;
        margin-left: auto;
        margin-right: auto;
        margin-top: 0;
        margin-bottom: 0;
        line-height: 1.55;
        text-align: center;
        text-wrap: balance;
    }

    @media (max-width: 768px) {
        .landing-header {
            padding: 14px 0 12px 0;
        }
        .landing-title {
            font-size: 1.7rem;
            line-height: 1.3;
            margin-bottom: 10px;
        }
        .landing-subtitle-container {
            max-width: 100%;
            padding: 0 16px;
        }
        .landing-subtitle {
            font-size: 0.94rem;
            line-height: 1.5;
        }
    }
    @media (max-width: 480px) {
        .landing-title {
            font-size: 1.4rem;
        }
        .landing-subtitle-container {
            padding: 0 10px;
        }
        .landing-subtitle {
            font-size: 0.88rem;
            line-height: 1.45;
        }
    }

    /* Upload Panel Card */
    .upload-panel-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px 28px;
        margin: 16px auto 26px auto;
        max-width: 820px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .upload-panel-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }
    .upload-panel-icon {
        font-size: 1.6rem;
        color: var(--accent-cyan);
    }
    .upload-panel-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .upload-panel-desc {
        font-size: 0.86rem;
        color: var(--text-secondary);
        margin-top: 2px;
    }
    .upload-specs-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        font-size: 0.78rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border-subtle);
        padding-top: 10px;
    }

    /* Feature Grid (Landing Page) */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        max-width: 960px;
        margin: 0 auto 30px auto;
    }
    .feature-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .feature-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
    }
    .feature-icon-badge {
        width: 34px;
        height: 34px;
        border-radius: var(--radius-sm);
        background-color: var(--bg-surface-elevated);
        border: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 12px;
        color: var(--accent-cyan);
    }
    .feature-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.42;
        margin: 0;
    }

    /* Metric / KPI Card Styling */
    [data-testid="stMetric"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
        transition: border-color 0.15s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-accent) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    /* Insight and Analytical Cards */
    .insight-box {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .insight-box-title {
        font-size: 0.94rem;
        font-weight: 600;
        color: var(--accent-cyan);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .insight-box-content {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Source Citation Badges */
    .source-tag {
        font-size: 0.74rem;
        color: var(--accent-cyan);
        background-color: var(--accent-cyan-glow);
        border: 1px solid var(--border-accent);
        padding: 2px 7px;
        border-radius: var(--radius-sm);
        margin-left: 6px;
        font-weight: 500;
        display: inline-block;
    }

    /* Streamlit Containers & Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stDownloadButton > button {
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
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

# Standard dark layout parameters for Plotly visualizations
DARK_PLOT_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        color="#F8FAFC",
        size=11,
    ),
    title=dict(font=dict(size=14, color="#F8FAFC")),
    xaxis=dict(
        title_font=dict(size=11, color="#94A3B8"),
        tickfont=dict(size=10, color="#94A3B8"),
        gridcolor="#1F2937",
        linecolor="#334155",
        zerolinecolor="#334155",
    ),
    yaxis=dict(
        title_font=dict(size=11, color="#94A3B8"),
        tickfont=dict(size=10, color="#94A3B8"),
        gridcolor="#1F2937",
        linecolor="#334155",
        zerolinecolor="#334155",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10, color="#94A3B8"),
        bgcolor="rgba(17, 24, 39, 0.85)",
        bordercolor="#1F2937",
        borderwidth=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1E293B",
        font_size=11,
        font_color="#F8FAFC",
        bordercolor="#334155",
    ),
    margin=dict(l=45, r=30, t=55, b=40),
)


def get_model_selection_index(
    selected_model: Optional[str],
    supported_models: Sequence[str],
) -> int:
    """Return the 0-based index of selected_model in supported_models.

    Falls back to 0 if selected_model is None or not present in supported_models.
    """
    if selected_model and selected_model in supported_models:
        return list(supported_models).index(selected_model)
    return 0


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
    if getattr(metric, "is_unavailable", False):
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
    if getattr(metric, "is_unavailable", False):
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
    """Render compact, enterprise-grade sidebar with branding, navigation, and model configuration."""
    with st.sidebar:
        # 1. Brand Section
        st.markdown(
            """
            <div class="sidebar-brand-container">
                <div class="sidebar-logo-badge">FS</div>
                <div>
                    <div class="sidebar-brand-title">FinSight</div>
                    <div class="sidebar-brand-sub">Financial Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        has_report = "extraction_report" in st.session_state and st.session_state.extraction_report is not None

        # 2. Section Navigation (When report is loaded)
        if has_report:
            st.markdown("<small style='color: #64748B; font-weight: 600; text-transform: uppercase;'>Navigation</small>", unsafe_allow_html=True)
            nav_options = [
                "📊 Overview",
                "📈 Visual Analytics",
                "📑 Financial Statements",
                "📐 Ratio Analysis",
                "💡 Executive Summary",
                "🤖 AI Financial Assistant",
                "📤 Export Reports",
            ]
            st.radio(
                "Section Selector",
                options=nav_options,
                index=0,
                key="app_navigation_section",
                label_visibility="collapsed",
            )
            st.markdown("---")

            # 3. Active Document Summary
            st.markdown("<small style='color: #64748B; font-weight: 600; text-transform: uppercase;'>Active Document</small>", unsafe_allow_html=True)
            parsed: ParsedPDF = st.session_state.get("parsed_pdf")
            rep: FinancialExtractionReport = st.session_state.get("extraction_report")
            filename = st.session_state.get("uploaded_filename", "Document.pdf")
            
            st.write(f"📄 **{filename}**")
            if rep and rep.company_name:
                st.write(f"🏢 **{rep.company_name}**")
            if parsed:
                st.caption(f"{parsed.total_pages} pages • {parsed.total_characters:,} characters")

            # Extraction status badge
            if rep:
                if getattr(rep, "is_incomplete", False):
                    st.markdown("<span class='status-pill-incomplete'>⚠️ Partial Extraction</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='status-pill-verified'>● Extraction Complete</span>", unsafe_allow_html=True)

            # Retrieval engine badge
            rag_pipeline: Optional[RAGPipeline] = st.session_state.get("rag_pipeline")
            if rag_pipeline:
                st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
                if not rag_pipeline.is_semantic_unavailable:
                    st.caption("🟢 **RAG Retrieval:** Semantic + BM25 Hybrid")
                else:
                    st.caption("🟡 **RAG Retrieval:** Keyword BM25 Fallback")

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset Analysis", use_container_width=True, help="Clear active session state and start a new document analysis"):
                clear_report_state()
                st.rerun()

            st.markdown("---")

        # 4. Model Configuration (Compact & Aligned)
        st.markdown("<small style='color: #64748B; font-weight: 600; text-transform: uppercase;'>Model Configuration</small>", unsafe_allow_html=True)
        default_api_key = getattr(config, "GEMINI_API_KEY", "") or ""
        supported_models = getattr(
            config,
            "SUPPORTED_GEMINI_MODELS",
            ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        )
        configured_model = getattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
        current_selected = st.session_state.get("selected_gemini_model") or configured_model
        default_idx = get_model_selection_index(current_selected, supported_models)

        api_key_input = st.text_input(
            "Gemini API Key",
            value=default_api_key,
            type="password",
            help="Configured in .env or override here.",
        )
        
        # Connection indicator
        if api_key_input:
            st.markdown("<div style='margin-top: -6px; margin-bottom: 8px;'><span class='status-pill-verified'>● Key Connected</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top: -6px; margin-bottom: 8px;'><span class='status-pill-incomplete'>⚠️ Key Required</span></div>", unsafe_allow_html=True)

        model_input = st.selectbox(
            "Primary Generative Model",
            options=supported_models,
            index=default_idx,
            help="Generative LLM used for multi-year statement extraction and report Q&A.",
        )
        st.session_state.selected_gemini_model = model_input

        fallback_model = getattr(config, "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
        st.caption(f"⚡ Fallback Model: `{fallback_model}`")

        # 5. Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.74rem; color: #64748B; line-height: 1.4;">
                <b>FinSight Platform v2.5</b><br>
                Autonomous Financial Statement Intelligence & Grounded RAG Analytics.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return api_key_input, model_input


@st.cache_data(show_spinner=False)
def get_cached_parsed_pdf(file_bytes: bytes, filename: str) -> ParsedPDF:
    """Parse PDF bytes once and cache the ParsedPDF object across runs."""
    pdf_stream = io.BytesIO(file_bytes)
    setattr(pdf_stream, "name", filename)
    return extract_text_from_pdf(pdf_stream)


@st.cache_data(show_spinner=False)
def get_cached_extraction_report(
    file_hash: str,
    api_key: Optional[str],
    model_name: str,
    _parsed_pdf: ParsedPDF,
    fallback_model_name: Optional[str] = None,
) -> FinancialExtractionReport:
    """Extract financial statements with Gemini and cache result by document hash and model."""
    effective_fallback = (
        fallback_model_name
        if fallback_model_name is not None
        else getattr(config, "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    )
    extractor = FinancialExtractor(
        api_key=api_key,
        model_name=model_name,
        fallback_model_name=effective_fallback,
        max_pages_per_section=4,
    )
    return extractor.extract(_parsed_pdf)


@st.cache_data(show_spinner=False)
def get_cached_executive_summary(
    file_hash: str,
    api_key: Optional[str],
    model_name: str,
    _report: FinancialExtractionReport,
    _ratios: List[FinancialRatios],
    _pipeline: Optional[RAGPipeline],
) -> ExecutiveSummary:
    """Generate AI executive summary and cache result by document hash and model."""
    fallback_model = getattr(config, "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    summary_gen = ExecutiveSummaryGenerator(
        api_key=api_key,
        model_name=model_name,
        fallback_model_name=fallback_model,
    )
    return summary_gen.generate_summary(
        report=_report,
        ratios=_ratios,
        pipeline=_pipeline,
    )


def run_pipeline(
    uploaded_file: Any,
    api_key: Optional[str],
    model_name: str,
    file_bytes: Optional[bytes] = None,
    file_hash: Optional[str] = None,
) -> bool:
    """Execute PDF parsing, LLM financial extraction, and ratio calculation."""
    if not api_key:
        st.error("⚠️ Gemini API Key is required to extract financial statements. Please configure it in the sidebar.")
        return False

    if file_bytes is None:
        file_bytes = extract_file_bytes(uploaded_file)
    if file_hash is None:
        file_hash = compute_pdf_hash(file_bytes)

    filename = getattr(uploaded_file, "name", "document.pdf")

    with st.status("Analyzing Financial Report...", expanded=True) as status:
        # Step 1: PDF Ingestion & Text Parsing (Cached)
        status.update(label="1/5: Parsing PDF pages with pdfplumber...")
        try:
            parsed_pdf = get_cached_parsed_pdf(file_bytes, filename)
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

        # Step 2: Gemini Structured Statement Extraction (Cached)
        status.update(label="2/5: Extracting structured financial statements with Gemini...")
        try:
            extraction_report = get_cached_extraction_report(
                file_hash=file_hash,
                api_key=api_key,
                model_name=model_name,
                _parsed_pdf=parsed_pdf,
            )
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

        # Step 4: Index Document in RAG Pipeline for Q&A (Cached)
        status.update(label="4/5: Indexing document passages for report Q&A...")
        rag_pipeline: Optional[RAGPipeline] = None
        try:
            rag_pipeline = get_or_create_rag_pipeline(
                parsed_pdf=parsed_pdf,
                api_key=api_key,
                file_hash=file_hash,
                doc_id=filename,
            )
            if rag_pipeline and rag_pipeline.is_semantic_unavailable:
                st.warning(
                    f"⚠️ Note: Semantic embedding generation is temporarily unavailable ({rag_pipeline.embedding_error}). "
                    "Document passages remain indexed using keyword retrieval (BM25)."
                )
        except Exception as e:
            st.warning(f"Note: RAG indexing encountered an issue: {e}")

        # Step 5: Generate AI Executive Summary (Cached)
        status.update(label="5/5: Generating AI Executive Summary...")
        exec_summary: Optional[ExecutiveSummary] = None
        try:
            exec_summary = get_cached_executive_summary(
                file_hash=file_hash,
                api_key=api_key,
                model_name=model_name,
                _report=extraction_report,
                _ratios=ratios_list,
                _pipeline=rag_pipeline,
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

        if getattr(extraction_report, "is_incomplete", False):
            status.update(
                label="Analysis incomplete: Model temporarily unavailable (partial data preserved)",
                state="error",
                expanded=True,
            )
        else:
            status.update(label="Analysis complete!", state="complete", expanded=False)

    return True


def render_header_banner(report: FinancialExtractionReport, current_section: str) -> None:
    """Render top enterprise header banner with company metadata and status indicator."""
    comp_name = report.company_name or "Financial Report Analysis"
    if report.annual_reports:
        years = [a.year for a in report.annual_reports]
        year_span = f"FY {min(years)} – FY {max(years)}"
        curr = report.annual_reports[0].currency or ""
        unit = report.annual_reports[0].unit or ""
        scale_tag = f"Currency: {curr} ({unit})" if curr or unit else ""
    else:
        year_span = "Fiscal Periods Not Identified"
        scale_tag = ""

    doc_name = st.session_state.get("uploaded_filename", "Corporate Report.pdf")
    is_inc = getattr(report, "is_incomplete", False)
    status_badge_html = (
        "<span class='status-pill-incomplete'>⚠️ Incomplete Extraction</span>"
        if is_inc
        else "<span class='status-pill-verified'>● Verified & Complete</span>"
    )

    meta_items = [f"<span class='header-meta-pill'>📅 {year_span}</span>"]
    if scale_tag:
        meta_items.append(f"<span class='header-meta-pill'>💱 {scale_tag}</span>")
    meta_items.append(f"<span class='header-meta-pill'>📑 {doc_name}</span>")
    meta_items.append(f"<span class='header-meta-pill'>🧭 {current_section}</span>")
    meta_html = "".join(meta_items)

    st.markdown(
        f"""
        <div class="header-banner-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 class="header-entity-name">🏢 {comp_name}</h1>
                    <div class="header-meta-row">
                        {meta_html}
                    </div>
                </div>
                <div>
                    {status_badge_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render compact, institutional-grade KPI cards for selected reporting year."""
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

    annual = next((a for a in annual_reports if a.year == selected_year), None)
    ratio_obj = next((r for r in ratios_list if r.year == selected_year), None)

    if not annual:
        return

    curr = annual.currency or ""
    unit = annual.unit or ""

    # Safe metric values
    rev_val = safe_metric_value(annual.income_statement, "revenue")
    np_val = safe_metric_value(annual.income_statement, "net_profit")
    assets_val = safe_metric_value(annual.balance_sheet, "total_assets")
    liab_val = safe_metric_value(annual.balance_sheet, "total_liabilities")

    # Ratio values
    growth_val = ratio_obj.revenue_growth_yoy_pct if ratio_obj else None
    gross_margin_val = ratio_obj.gross_margin_pct if ratio_obj else None
    net_margin_val = ratio_obj.net_profit_margin_pct if ratio_obj else None
    de_val = ratio_obj.debt_to_equity if ratio_obj else None
    roe_val = ratio_obj.roe if ratio_obj else None

    # First Row: Core Scale & Financial Position (Revenue, Net Profit, Total Assets, Total Liabilities)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rev_display = format_currency_value(rev_val, curr, unit)
        growth_display = f"{growth_val:+.1f}% YoY" if growth_val is not None else None
        st.metric(
            label="Total Revenue",
            value=rev_display,
            delta=growth_display,
            help="Consolidated revenue or net sales from operations for the fiscal year.",
        )
        if rev_val is None:
            st.caption("⚠️ Revenue not reported.")

    with col2:
        np_display = format_currency_value(np_val, curr, unit)
        st.metric(
            label="Net Profit (Loss)",
            value=np_display,
            help="Consolidated net profit/loss after tax for the fiscal year.",
        )
        if np_val is None:
            st.caption("⚠️ Net Profit not reported.")
        elif np_val < 0:
            st.caption("🔻 Net loss reported.")

    with col3:
        assets_display = format_currency_value(assets_val, curr, unit)
        st.metric(
            label="Total Assets",
            value=assets_display,
            help="Total economic resources owned by the entity on the Balance Sheet.",
        )
        if assets_val is None:
            st.caption("⚠️ Total Assets not reported.")

    with col4:
        liab_display = format_currency_value(liab_val, curr, unit)
        st.metric(
            label="Total Liabilities",
            value=liab_display,
            help="Total financial obligations on the Balance Sheet.",
        )
        if liab_val is None:
            st.caption("⚠️ Total Liabilities not reported.")

    # Second Row: Margins, Leverage & Solvency
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        gm_display = f"{gross_margin_val:.2f}%" if gross_margin_val is not None else "Unavailable"
        st.metric(
            label="Gross Margin",
            value=gm_display,
            help="(Revenue - COGS) / Revenue × 100",
        )
        if gross_margin_val is None:
            st.caption("ℹ️ COGS not reported.")

    with col6:
        margin_display = f"{net_margin_val:.2f}%" if net_margin_val is not None else "Unavailable"
        st.metric(
            label="Net Profit Margin",
            value=margin_display,
            help="Net Profit / Revenue × 100",
        )
        if net_margin_val is None:
            st.caption("⚠️ Requires Net Profit & Revenue.")

    with col7:
        de_display = f"{de_val:.2f}x" if de_val is not None else "Unavailable"
        st.metric(
            label="Debt-to-Equity Ratio",
            value=de_display,
            help="Total Debt (Borrowings) / Shareholders' Equity.",
        )
        if de_val is None:
            st.caption("ℹ️ Total Debt not reported.")

    with col8:
        roe_display = f"{roe_val:.2f}%" if roe_val is not None else "Unavailable"
        st.metric(
            label="Return on Equity (ROE)",
            value=roe_display,
            help="Net Profit / Shareholders' Equity × 100",
        )
        if roe_val is None:
            st.caption("⚠️ Requires Net Profit & Equity.")


def render_charts(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render interactive Plotly dark charts with negative value support and gap preservation."""
    st.subheader("Financial Performance & Trajectory Analytics")

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
            "Operating Profit": safe_metric_value(ann.income_statement, "operating_profit"),
            "Net Profit": safe_metric_value(ann.income_statement, "net_profit"),
            "Total Assets": safe_metric_value(ann.balance_sheet, "total_assets"),
            "Total Liabilities": safe_metric_value(ann.balance_sheet, "total_liabilities"),
            "Shareholders Equity": safe_metric_value(ann.balance_sheet, "equity"),
            "Operating Cash Flow": safe_metric_value(ann.cash_flow_statement, "operating_cash_flow"),
            "CapEx": safe_metric_value(ann.cash_flow_statement, "capital_expenditure"),
            "Gross Margin %": r.gross_margin_pct if r else None,
            "Net Profit Margin %": r.net_profit_margin_pct if r else None,
            "Revenue Growth YoY %": r.revenue_growth_yoy_pct if r else None,
            "ROE %": r.roe if r else None,
            "Currency": ann.currency or "",
            "Unit": ann.unit or "",
        })

    df = pd.DataFrame(records)

    tab_overview, tab_balance, tab_margins, tab_growth = st.tabs([
        "📊 Revenue & Profitability Trends",
        "⚖️ Balance Sheet & Capital Structure",
        "📈 Financial Ratio Comparisons",
        "🚀 YoY Growth Dynamics & Cash Flow",
    ])

    unit_label = f" ({df['Unit'].iloc[0]})" if not df.empty and df["Unit"].iloc[0] else ""
    curr_label = f" in {df['Currency'].iloc[0]}" if not df.empty and df["Currency"].iloc[0] else ""

    # Tab 1: Revenue & Profitability
    with tab_overview:
        has_rev = df["Revenue"].notna().any()
        has_ebit = df["Operating Profit"].notna().any()
        has_np = df["Net Profit"].notna().any()

        if not has_rev and not has_np and not has_ebit:
            st.info("No revenue, operating profit, or net profit data available across reporting years to plot.")
        else:
            fig = go.Figure()
            if has_rev:
                fig.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Revenue"],
                        name="Revenue",
                        marker_color="#38BDF8",  # Electric Cyan
                        text=df["Revenue"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )
            if has_ebit:
                fig.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Operating Profit"],
                        name="Operating Profit (EBIT)",
                        marker_color="#0284C7",  # Deep Sky Blue
                        text=df["Operating Profit"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )
            if has_np:
                # Color code negative net profit
                colors = ["#10B981" if (pd.notna(v) and v >= 0) else "#EF4444" for v in df["Net Profit"]]
                fig.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Net Profit"],
                        name="Net Profit (Loss)",
                        marker_color=colors,
                        text=df["Net Profit"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )

            fig.add_hline(y=0, line_dash="solid", line_color="#334155", line_width=1)

            layout_args = dict(DARK_PLOT_LAYOUT)
            layout_args.update(
                barmode="group",
                title=dict(text=f"Revenue, Operating Profit & Net Profit Trajectory{curr_label}{unit_label}", font=dict(size=14, color="#F8FAFC")),
                xaxis_title="Fiscal Year",
                yaxis_title=f"Amount{unit_label}",
            )
            fig.update_layout(**layout_args)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Balance Sheet & Capital Structure
    with tab_balance:
        has_assets = df["Total Assets"].notna().any()
        has_liab = df["Total Liabilities"].notna().any()
        has_eq = df["Shareholders Equity"].notna().any()

        if not has_assets and not has_liab and not has_eq:
            st.info("No balance sheet assets or liabilities data available to plot.")
        else:
            fig_bs = go.Figure()
            if has_assets:
                fig_bs.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Total Assets"],
                        name="Total Assets",
                        marker_color="#2563EB",  # Royal Cobalt
                        text=df["Total Assets"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )
            if has_liab:
                fig_bs.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Total Liabilities"],
                        name="Total Liabilities",
                        marker_color="#F97316",  # Orange Amber
                        text=df["Total Liabilities"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )
            if has_eq:
                fig_bs.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Shareholders Equity"],
                        name="Shareholders' Equity",
                        marker_color="#10B981",  # Emerald Green
                        text=df["Shareholders Equity"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                        textposition="auto",
                    )
                )

            layout_bs = dict(DARK_PLOT_LAYOUT)
            layout_bs.update(
                barmode="group",
                title=dict(text=f"Balance Sheet Capital Structure: Assets vs. Liabilities vs. Equity{curr_label}{unit_label}", font=dict(size=14, color="#F8FAFC")),
                xaxis_title="Fiscal Year",
                yaxis_title=f"Amount{unit_label}",
            )
            fig_bs.update_layout(**layout_bs)
            st.plotly_chart(fig_bs, use_container_width=True)

    # Tab 3: Financial Ratio Comparisons
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
                        line=dict(color="#06B6D4", width=3),  # Cyan
                        marker=dict(size=8),
                    )
                )
            if has_margin:
                fig_margin.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["Net Profit Margin %"],
                        name="Net Profit Margin (%)",
                        mode="lines+markers",
                        line=dict(color="#A78BFA", width=3),  # Soft Purple
                        marker=dict(size=8),
                    )
                )
            if has_roe:
                fig_margin.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["ROE %"],
                        name="Return on Equity (%)",
                        mode="lines+markers",
                        line=dict(color="#F59E0B", width=2, dash="dash"),  # Amber
                        marker=dict(size=8),
                    )
                )

            fig_margin.add_hline(y=0, line_dash="dot", line_color="#334155")

            layout_margin = dict(DARK_PLOT_LAYOUT)
            layout_margin.update(
                title=dict(text="Profitability Margins & Return on Equity Trends (%)", font=dict(size=14, color="#F8FAFC")),
                xaxis_title="Fiscal Year",
                yaxis_title="Percentage (%)",
            )
            fig_margin.update_layout(**layout_margin)
            st.plotly_chart(fig_margin, use_container_width=True)

    # Tab 4: YoY Growth Dynamics & Cash Flow
    with tab_growth:
        c_left, c_right = st.columns(2)

        with c_left:
            has_growth = df["Revenue Growth YoY %"].notna().any()
            if not has_growth:
                st.info("YoY Revenue Growth requires at least two consecutive reporting fiscal years with reported revenue.")
            else:
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

                fig_growth.add_hline(y=0, line_dash="solid", line_color="#334155", line_width=1)

                layout_growth = dict(DARK_PLOT_LAYOUT)
                layout_growth.update(
                    title=dict(text="YoY Revenue Growth Rate Dynamics (%)", font=dict(size=14, color="#F8FAFC")),
                    xaxis_title="Fiscal Year",
                    yaxis_title="Growth Rate (%)",
                )
                fig_growth.update_layout(**layout_growth)
                st.plotly_chart(fig_growth, use_container_width=True)

        with c_right:
            has_cf = df["Operating Cash Flow"].notna().any()
            has_capex = df["CapEx"].notna().any()
            if not has_cf and not has_capex:
                st.info("Cash Flow Statement items (Operating Cash Flow / CapEx) not reported in extracted periods.")
            else:
                fig_cf = go.Figure()
                if has_cf:
                    fig_cf.add_trace(
                        go.Bar(
                            x=df["Year"],
                            y=df["Operating Cash Flow"],
                            name="Operating Cash Flow",
                            marker_color="#10B981",
                            text=df["Operating Cash Flow"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                            textposition="auto",
                        )
                    )
                if has_capex:
                    fig_cf.add_trace(
                        go.Bar(
                            x=df["Year"],
                            y=df["CapEx"],
                            name="Capital Expenditure (CapEx)",
                            marker_color="#F59E0B",
                            text=df["CapEx"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else ""),
                            textposition="auto",
                        )
                    )

                layout_cf = dict(DARK_PLOT_LAYOUT)
                layout_cf.update(
                    barmode="group",
                    title=dict(text=f"Cash Flow & Capital Allocation{curr_label}{unit_label}", font=dict(size=14, color="#F8FAFC")),
                    xaxis_title="Fiscal Year",
                    yaxis_title=f"Amount{unit_label}",
                )
                fig_cf.update_layout(**layout_cf)
                st.plotly_chart(fig_cf, use_container_width=True)


def render_financial_tables(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render structured financial statement tables grouped by reporting year with source audit tags."""
    st.subheader("Audited Financial Statements & Statement Tables")
    st.caption("Standardized financial statements extracted directly from report tables with explicit page audit citations.")

    if not annual_reports:
        st.info("No financial statements extracted.")
        return

    years = [a.year for a in annual_reports]
    ratio_dict: Dict[int, FinancialRatios] = {r.year: r for r in ratios_list}

    tab_summary, tab_is, tab_bs, tab_cf = st.tabs([
        "📋 Consolidated Overview",
        "💵 Income Statement",
        "⚖️ Balance Sheet",
        "🔄 Cash Flow Statement",
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

    # Dedicated CSV export action for financial statements
    st.markdown("---")
    report_obj = st.session_state.get("extraction_report")
    if report_obj:
        safe_entity = re.sub(r"[^a-zA-Z0-9_\-]", "_", (report_obj.company_name or "financial").strip().lower())
        csv_payload = generate_consolidated_csv(report_obj, ratios_list)
        st.download_button(
            label="📥 Download Financial Statements (CSV)",
            data=csv_payload,
            file_name=f"{safe_entity}_financial_statements.csv",
            mime="text/csv",
            key="dl_financial_data_table_btn",
            help="Download complete statements and ratios as a consolidated CSV spreadsheet.",
            use_container_width=True,
        )


def render_ratios_section(
    annual_reports: List[AnnualFinancialData],
    ratios_list: List[FinancialRatios],
) -> None:
    """Render dedicated Financial Ratios section with analytical benchmark cards."""
    st.subheader("Financial Ratios & Structural Analytics")
    st.caption("Pure-Python computed financial performance benchmarks, profitability margins, and capital solvency.")

    if not annual_reports:
        st.info("No financial ratios available.")
        return

    years = [a.year for a in annual_reports]
    ratio_dict: Dict[int, FinancialRatios] = {r.year: r for r in ratios_list}

    # Summary table
    ratio_rows = [
        ("Gross Margin (%)", lambda r: format_pct_value(r.gross_margin_pct), "Gross Profit / Revenue × 100"),
        ("Net Profit Margin (%)", lambda r: format_pct_value(r.net_profit_margin_pct), "Net Income / Revenue × 100"),
        ("Return on Equity (ROE) (%)", lambda r: format_pct_value(r.roe), "Net Income / Shareholders' Equity × 100"),
        ("Debt-to-Equity (x)", lambda r: format_ratio_value(r.debt_to_equity), "Total Debt / Shareholders' Equity"),
        ("YoY Revenue Growth (%)", lambda r: format_pct_value(r.revenue_growth_yoy_pct), "YoY Revenue Change"),
    ]

    ratio_table = []
    for name, formatter, formula in ratio_rows:
        row = {"Ratio Metric": name, "Formula": formula}
        for y in years:
            r_obj = ratio_dict.get(y)
            row[f"FY {y}"] = formatter(r_obj) if r_obj else "Unavailable"
        ratio_table.append(row)

    st.dataframe(pd.DataFrame(ratio_table).set_index("Ratio Metric"), use_container_width=True)

    st.markdown("---")
    st.subheader("Institutional Benchmark Definitions & Interpretation")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-box-title">📈 Profitability & Margin Dynamics</div>
                <div class="insight-box-content">
                    <b>Gross Margin:</b> Measures fundamental production efficiency and pricing power before general corporate overhead. Missing when COGS is not broken down.<br><br>
                    <b>Net Profit Margin:</b> Reflects comprehensive bottom-line conversion after operating expenses, finance costs, and taxation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="insight-box">
                <div class="insight-box-title">⚖️ Capital Structure & Solvency Health</div>
                <div class="insight-box-content">
                    <b>Debt-to-Equity Ratio:</b> Evaluates financial leverage by comparing explicit interest-bearing borrowings directly to net worth. Total liabilities are strictly separated.<br><br>
                    <b>Return on Equity (ROE):</b> Measures the efficiency with which shareholders' equity capital is converted into net earnings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


RAG_PIPELINE_CACHE_VERSION = "v3_header_context_1200"


@st.cache_resource(show_spinner=False)
def get_cached_rag_pipeline(
    file_hash: str,
    _parsed_pdf: ParsedPDF,
    api_key: Optional[str] = None,
    doc_id: Optional[str] = None,
    cache_version: str = RAG_PIPELINE_CACHE_VERSION,
) -> RAGPipeline:
    """Create and cache the RAGPipeline with indexed document chunks and embeddings."""
    pipeline = create_rag_pipeline(
        api_key=api_key,
        retriever_type="hybrid" if api_key else "bm25",
        chunk_size=1200,
        chunk_overlap=300,
    )
    pipeline.ingest(_parsed_pdf, doc_id=doc_id)
    return pipeline


def get_or_create_rag_pipeline(
    parsed_pdf: Optional[ParsedPDF],
    api_key: Optional[str] = None,
    file_hash: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Optional[RAGPipeline]:
    """Retrieve existing RAG pipeline from session state or cached resource."""
    if "rag_pipeline" in st.session_state and st.session_state.rag_pipeline is not None:
        if getattr(st.session_state.rag_pipeline, "_version", None) == RAG_PIPELINE_CACHE_VERSION:
            return st.session_state.rag_pipeline
    if parsed_pdf is not None and parsed_pdf.pages:
        effective_hash = file_hash or st.session_state.get("uploaded_file_hash") or "default_doc"
        effective_doc_id = doc_id or st.session_state.get("uploaded_filename", "document.pdf")
        effective_api_key = api_key or getattr(config, "GEMINI_API_KEY", None)
        pipeline = get_cached_rag_pipeline(
            file_hash=effective_hash,
            _parsed_pdf=parsed_pdf,
            api_key=effective_api_key,
            doc_id=effective_doc_id,
            cache_version=RAG_PIPELINE_CACHE_VERSION,
        )
        setattr(pipeline, "_version", RAG_PIPELINE_CACHE_VERSION)
        st.session_state.rag_pipeline = pipeline
        return pipeline
    return None


def render_qa_chat(api_key: Optional[str], model_name: str) -> None:
    """Render interactive report-grounded Q&A chat section with dark theme aesthetics."""
    st.subheader("🤖 AI Financial Research Assistant")
    st.caption(
        "Ask questions regarding revenue segments, enterprise risks, capital expenditures, or accounting disclosures. "
        "Answers are strictly grounded in retrieved passages with verifiable page citations."
    )

    # Initialize chat history if not present
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    parsed_pdf = st.session_state.get("parsed_pdf")
    file_hash = st.session_state.get("uploaded_file_hash")
    doc_filename = st.session_state.get("uploaded_filename")
    pipeline = get_or_create_rag_pipeline(parsed_pdf, api_key=api_key, file_hash=file_hash, doc_id=doc_filename)

    # Status / Info and clear conversation
    col_info, col_clear = st.columns([4, 1])
    with col_info:
        if pipeline and pipeline.chunk_count > 0:
            sem_tag = " (Semantic + BM25 Hybrid)" if not pipeline.is_semantic_unavailable else " (Keyword BM25 Only)"
            st.markdown(
                f"<small style='color: #94A3B8;'>Indexed <b>{pipeline.chunk_count}</b> passages across <b>{len(pipeline.indexed_pages)}</b> pages ready for Q&A{sem_tag}.</small>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No report text is currently indexed for Q&A.")
            return

    if pipeline and pipeline.is_semantic_unavailable:
        st.warning(
            "⚠️ **Semantic Retrieval Notice:** Gemini embedding generation is temporarily unavailable (quota limit or API error). "
            "Keyword retrieval (BM25) is actively being used for document Q&A."
        )

    with col_clear:
        if st.session_state.qa_messages:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
                st.session_state.qa_messages = []
                st.rerun()

    # Suggested prompt pills for convenience
    st.markdown("<small style='color: #64748B; font-weight: 600;'>Suggested Prompts:</small>", unsafe_allow_html=True)
    suggested_cols = st.columns(4)
    suggested_questions = [
        "What was the total revenue?",
        "What are the principal risk factors?",
        "What was operating cash flow?",
        "What was the capital expenditure?",
    ]
    query_from_suggestion = None
    for idx, (col, q_text) in enumerate(zip(suggested_cols, suggested_questions)):
        with col:
            if st.button(f"💡 {q_text}", key=f"sug_q_{idx}", use_container_width=True):
                query_from_suggestion = q_text

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("has_insufficient_evidence"):
                st.info("ℹ️ The uploaded report does not contain enough information to fully answer this question.")

            # Display cited pages badges
            cited = msg.get("cited_pages")
            if cited:
                citations_str = " ".join([f"<span class='source-tag'>Page {p}</span>" for p in cited])
                st.markdown(f"<div style='margin-bottom: 8px;'><small><b>Sources:</b> {citations_str}</small></div>", unsafe_allow_html=True)

            # Display source citations expander if available
            sources = msg.get("sources")
            if sources:
                with st.expander(f"📄 Retrieved Sources ({len(sources)} passages)", expanded=False):
                    for idx, src in enumerate(sources):
                        st.markdown(f"**[Page {src['page_number']}]** (Score: `{src['score']:.4f}`):")
                        st.markdown(f"> *{src['text']}*")
                        if idx < len(sources) - 1:
                            st.markdown("---")

    # Chat input
    user_query = st.chat_input("Ask a question about this report (e.g., 'What are the principal risk factors?')...")
    effective_query = user_query or query_from_suggestion

    if effective_query:
        if not api_key:
            st.error("⚠️ Gemini API key is required to answer questions. Please configure it in the sidebar.")
            return

        st.session_state.qa_messages.append({"role": "user", "content": effective_query})

        with st.spinner("Searching report passages and generating grounded answer..."):
            fallback_model = getattr(config, "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
            qa_engine = ReportQAEngine(
                api_key=api_key,
                model_name=model_name,
                fallback_model_name=fallback_model,
                top_k=6,
            )
            qa_result = qa_engine.answer_question(effective_query, pipeline)

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
                "model_used": qa_result.model_used,
            })

        st.rerun()


def render_executive_summary_section(
    report: FinancialExtractionReport,
    ratios: List[FinancialRatios],
    api_key: Optional[str],
    model_name: str,
) -> None:
    """Render structured AI Executive Financial Insights briefing with dark card styling."""
    st.subheader("💡 AI Executive Financial Intelligence Briefing")

    cached_summary: Optional[ExecutiveSummary] = st.session_state.get("executive_summary")

    # Header control row: Metadata on left, Regenerate / Generate button on right
    col_meta, col_btn = st.columns([4, 1])

    with col_meta:
        if cached_summary and not cached_summary.error:
            cited_str = f" • Sources: {', '.join(f'Page {p}' for p in cached_summary.cited_pages)}" if cached_summary.cited_pages else ""
            st.caption(f"🤖 Synthesized by **{cached_summary.model_used or model_name}** at {cached_summary.generated_at}{cited_str}")
        else:
            st.caption("AI strategic briefing combining verified statement data, calculated ratios, and reported management disclosures.")

    with col_btn:
        regen_label = "🔄 Regenerate" if cached_summary else "✨ Generate Briefing"
        if st.button(regen_label, key="gen_summary_btn", use_container_width=True, type="primary"):
            if not api_key:
                st.error("⚠️ Gemini API key is required to generate the executive summary. Please configure it in the sidebar.")
                return
            with st.spinner("Analyzing multi-year performance and synthesizing executive briefing..."):
                get_cached_executive_summary.clear()
                pipeline = get_or_create_rag_pipeline(
                    st.session_state.get("parsed_pdf"),
                    api_key=api_key,
                    file_hash=st.session_state.get("uploaded_file_hash"),
                    doc_id=st.session_state.get("uploaded_filename"),
                )
                fallback_model = getattr(config, "GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
                generator = ExecutiveSummaryGenerator(
                    api_key=api_key,
                    model_name=model_name,
                    fallback_model_name=fallback_model,
                )
                st.session_state.executive_summary = generator.generate_summary(report, ratios, pipeline)
            st.rerun()

    # Display cached summary
    cached_summary = st.session_state.get("executive_summary")
    if cached_summary:
        if getattr(cached_summary, "is_incomplete", False):
            st.warning(
                "⚠️ **Partial Briefing Notice:** This executive summary was synthesized from an incomplete financial "
                "report extraction due to temporary model availability limits. Observations reflect only successfully extracted sections."
            )
        if cached_summary.error:
            st.warning(f"Executive Summary Notice: {cached_summary.error}")

        with st.container(border=True):
            st.markdown(cached_summary.summary_text)
    else:
        st.info("Click **✨ Generate Briefing** above to synthesize an AI executive financial briefing for this report.")


def render_audit_and_validation(report: FinancialExtractionReport) -> None:
    """Render audit notices, missing fields breakdown, and extraction warnings."""
    with st.expander("🔍 Extraction & Statement Audit Details", expanded=False):
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
                    st.write(f"- **FY {a.year}:** All standard statement figures reported.")

        if report.extraction_notes:
            st.markdown("---")
            st.markdown("**Extraction Notes & Accounting Consistency Checks:**")
            for note in report.extraction_notes:
                st.caption(f"• {note}")


def render_export_section(
    report: FinancialExtractionReport,
    ratios: List[FinancialRatios],
    summary: Optional[ExecutiveSummary],
) -> None:
    """Render dedicated Export Reports section with file downloads and audit verification."""
    st.subheader("📤 Export Financial Intelligence Reports")
    st.caption("Generate publication-ready executive reports and structured CSV datasets with strict data provenance.")

    comp_name = report.company_name or "Financial Report Analysis"
    safe_entity = re.sub(r"[^a-zA-Z0-9_\-]", "_", comp_name.strip().lower())

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.container(border=True):
            st.markdown("### 📥 Consolidated Financial Data (CSV)")
            st.markdown(
                "Export complete multi-year income statements, balance sheets, cash flow metrics, and pure-Python "
                "calculated financial ratios into a structured spreadsheet. Includes explicit source page citations for every line item."
            )
            csv_payload = generate_consolidated_csv(report, ratios)
            st.download_button(
                label="Download CSV Dataset",
                data=csv_payload,
                file_name=f"{safe_entity}_financial_data.csv",
                mime="text/csv",
                key="export_page_csv_btn",
                type="primary",
                use_container_width=True,
            )

    with exp_col2:
        with st.container(border=True):
            st.markdown("### 📄 Executive Financial Report (PDF)")
            st.markdown(
                "Generate a professional, publication-quality executive briefing document with ReportLab vector trajectory charts, "
                "consolidated statement tables, audit citations, and dynamic accounting equation verification."
            )
            pdf_payload = generate_executive_pdf(
                report=report,
                ratios=ratios,
                summary=summary,
            )
            st.download_button(
                label="Download Executive PDF",
                data=pdf_payload,
                file_name=f"{safe_entity}_executive_report.pdf",
                mime="application/pdf",
                key="export_page_pdf_btn",
                type="primary",
                use_container_width=True,
            )

    st.markdown("---")
    render_audit_and_validation(report)


def render_landing_page(api_key: Optional[str], model_name: str) -> None:
    """Render the professional upload-focused welcome screen when no report is loaded."""
    st.markdown(
        """
        <div class="landing-header">
            <div class="landing-brand-tag">FinSight Intelligence Platform</div>
            <h1 class="landing-title">Institutional Financial <span>Statement Intelligence</span></h1>
            <div class="landing-subtitle-container">
                <p class="landing-subtitle">
                    Transform annual reports and Form 10-K filings into audited multi-year statements,
                    solvency benchmarks, and grounded AI insights.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main prominent PDF upload panel
    with st.container():
        st.markdown(
            """
            <div class="upload-panel-card">
                <div class="upload-panel-header">
                    <div class="upload-panel-icon">📄</div>
                    <div>
                        <div class="upload-panel-title">Upload Financial Report</div>
                        <div class="upload-panel-desc">Upload an audited annual report or Form 10-K PDF to begin analysis.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload Annual Report or Form 10-K (PDF)",
            type=["pdf"],
            help="Upload a corporate financial report or 10-K document (PDF format).",
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="upload-specs-row" style="max-width: 820px; margin: -10px auto 20px auto;">
                <span>Supported format: <b>PDF</b></span>
                <span>Document types: <b>Audited Annual Reports & 10-K</b></span>
                <span>Max size: <b>200 MB</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # If user selected a file on the landing page, trigger the analysis button
    if uploaded_file is not None:
        file_bytes = extract_file_bytes(uploaded_file)
        file_hash = compute_pdf_hash(file_bytes)
        file_size_mb = len(file_bytes) / (1024 * 1024)

        col_left, col_btn, col_right = st.columns([1, 2, 1])
        with col_btn:
            st.info(f"📁 Selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
            if st.button("🚀 Start Financial Analysis", type="primary", use_container_width=True):
                success = run_pipeline(
                    uploaded_file=uploaded_file,
                    api_key=api_key,
                    model_name=model_name,
                    file_bytes=file_bytes,
                    file_hash=file_hash,
                )
                if success:
                    st.rerun()

    # Four-section feature overview below the upload area
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon-badge">📑</div>
                <div class="feature-title">Financial Data Extraction</div>
                <div class="feature-desc">Multi-year statement parsing across Balance Sheet, Income Statement, and Cash Flow with audited page citations.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon-badge">📐</div>
                <div class="feature-title">Intelligent Ratio Analysis</div>
                <div class="feature-desc">Pure-Python calculation of gross margin, net margin, debt-to-equity, ROE, and consecutive YoY growth dynamics.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon-badge">🤖</div>
                <div class="feature-title">AI-Powered Financial Q&A</div>
                <div class="feature-desc">Strictly grounded RAG retrieval across corporate footnotes, risk factors, and accounting disclosures with exact page citations.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon-badge">📄</div>
                <div class="feature-title">Executive Report Generation</div>
                <div class="feature-desc">Publication-ready vector PDF executive briefings and comprehensive consolidated statement CSV spreadsheets.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main Streamlit execution routine."""
    api_key, model_name = render_sidebar()

    # Determine current navigation section
    current_section = st.session_state.get("app_navigation_section", "📊 Overview")

    # If no report has been extracted yet, show the streamlined landing page
    has_active_report = (
        "extraction_report" in st.session_state
        and st.session_state.extraction_report is not None
        and st.session_state.get("uploaded_file_hash") is not None
    )

    if not has_active_report:
        render_landing_page(api_key, model_name)
        return

    # Once a document is loaded and analyzed, render the enterprise dashboard
    report: FinancialExtractionReport = st.session_state.extraction_report

    # Ensure all annual reports conform to schema
    for ann in report.annual_reports:
        sanitize_and_flag_annual_data(ann)

    if "ratios" not in st.session_state or not st.session_state.ratios:
        st.session_state.ratios = calculate_report_ratios(report)

    ratios: List[FinancialRatios] = st.session_state.get("ratios", [])

    if not report.annual_reports:
        if getattr(report, "is_incomplete", False):
            st.warning(
                "⚠️ **Extraction Incomplete:** The Gemini model was temporarily unavailable. "
                "No financial statements could be extracted before pausing. Please retry below."
            )
        else:
            st.warning(
                "No structured financial statements could be extracted from this document. "
                "The report might be an image-only scan or does not contain standard financial statement tables."
            )
        render_audit_and_validation(report)
        return

    # Incomplete extraction recovery banner
    if getattr(report, "is_incomplete", False):
        st.warning(
            "⚠️ **Partial Extraction Notice:** High demand or transient availability errors interrupted extraction. "
            "Successfully extracted figures are displayed below. You can retry to complete missing sections."
        )

    # Render Top Header Banner with Company Name, Period, and Status
    render_header_banner(report, current_section)

    # Multi-page Section Router (Enterprise SaaS Navigation)
    if current_section == "📊 Overview":
        # 1. Top-Level KPI Cards Grid
        render_kpis(report.annual_reports, ratios)
        st.markdown("---")

        # 2. Executive Highlights Preview
        cached_summary = st.session_state.get("executive_summary")
        if cached_summary and not cached_summary.error:
            st.markdown(
                f"""
                <div class="insight-box">
                    <div class="insight-box-title">💡 Strategic Executive Briefing</div>
                    <div class="insight-box-content">{cached_summary.summary_text[:650]}... <i>(Navigate to 'Executive Summary' for full briefing)</i></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")

        # 3. Interactive Plotly Trajectory Visualizations
        render_charts(report.annual_reports, ratios)

    elif current_section == "📈 Visual Analytics":
        render_charts(report.annual_reports, ratios)

    elif current_section == "📑 Financial Statements":
        render_financial_tables(report.annual_reports, ratios)

    elif current_section == "📐 Ratio Analysis":
        render_ratios_section(report.annual_reports, ratios)

    elif current_section == "💡 Executive Summary":
        render_executive_summary_section(report, ratios, api_key, model_name)

    elif current_section == "🤖 AI Financial Assistant":
        render_qa_chat(api_key, model_name)

    elif current_section == "📤 Export Reports":
        render_export_section(report, ratios, st.session_state.get("executive_summary"))


if __name__ == "__main__":
    main()
