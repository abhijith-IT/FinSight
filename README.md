
# FinSight - Financial Statement Intelligence

FinSight is an AI-powered financial intelligence dashboard that extracts multi-year corporate financial statements from annual report PDFs (or 10-K filings), computes verified financial ratios, provides interactive trend visualizations, enables report-grounded Q&A with source page citations, and generates executive briefing summaries.

---
## Key Features

1. **Structured PDF Ingestion & Parsing**
   - High-precision page-by-page text extraction powered by `pdfplumber`.
   - Document change detection using SHA-256 byte hashing to guarantee report isolation and prevent stale PDF data caching.

2. **Gemini Multi-Year Statement Extraction**
   - Extracts multi-year Income Statements, Balance Sheets, and Cash Flow Statements into structured Pydantic models.
   - Extracts explicit Cost of Goods Sold (COGS) and Total Debt without inferring or confusing with Total Liabilities.
   - Preserves source page attribution for every financial figure.

3. **Deterministic Financial Ratio Engine**
   - Pure Python calculations:
     - **Gross Margin %** = `(Revenue - Cost of Goods Sold) / Revenue * 100`
     - **Net Profit Margin %** = `Net Profit / Revenue * 100`
     - **Return on Equity (ROE) %** = `Net Profit / Shareholders' Equity * 100`
     - **Debt-to-Equity** = `Total Debt / Shareholders' Equity`
     - **YoY Revenue Growth %** = `(Current Revenue - Previous Revenue) / Previous Revenue * 100` (consecutive reporting years)
   - Resilient against missing metrics, zero denominators, and non-consecutive years (returns unavailable/null instead of crashing or inventing values).

4. **Interactive Streamlit Dashboard**
   - KPI cards for key margins, leverage, and return metrics.
   - Plotly interactive charts for revenue trends, net profit, margins, and growth.
   - Formatted multi-year financial statements tables with source page audit tags.

5. **Report-Grounded Q&A (RAG)**
   - Modular BM25 passage retrieval preserving 1-indexed page numbers.
   - Strictly grounded Gemini answers citing source pages (`[Page X]`).
   - Automated detection of insufficient evidence when queries cannot be answered from the report text.

6. **AI Executive Summary**
   - Synthesizes multi-year performance, profitability trends, financial health, and management disclosures with verified citations.
   - Automated deterministic fallback briefing if LLM generation is unavailable.

---

## Project Structure

```
FinSight/
├── analytics/
│   └── ratios.py              # Pure Python financial ratio engine
├── extraction/
│   ├── financial_schema.py    # Pydantic schemas (Income Statement, Balance Sheet, Cash Flow)
│   ├── extractor.py           # Gemini-based structured financial statement extractor
│   └── validation.py          # Data validation and accounting consistency checks
├── generation/
│   └── summary.py             # AI Executive Summary briefing generator
├── ingestion/
│   └── pdf_parser.py          # PDF text extraction and page structuring
├── rag/
│   ├── chunker.py             # Document chunking with page attribution
│   ├── pipeline.py            # RAG pipeline coordinator
│   ├── qa.py                  # Grounded Q&A engine with citation reconciliation
│   ├── retriever.py           # Pluggable retrievers (BM25, Vector)
│   └── schema.py              # DocumentChunk and RetrievalResult schemas
├── tests/                     # Comprehensive unit test suite (71 tests)
├── app.py                     # Streamlit web application
├── config.py                  # Environment configuration
├── requirements.txt           # Python package dependencies
├── .env.example               # Example environment variables template
└── README.md                  # Project documentation
```

---

## Prerequisites

- Python 3.10+
- A Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

---

## Installation & Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd D:\Elevate\FinSight
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install runtime dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and insert your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
   *(Note: You can also enter or override the Gemini API Key directly in the Streamlit sidebar at runtime).*

---

## Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Usage Workflow:
1. Ensure your Gemini API Key is configured in `.env` or entered in the sidebar.
2. Upload a corporate annual report or 10-K in PDF format.
3. Click **🚀 Analyze Report**.
4. Explore extracted statements, KPI cards, interactive Plotly charts, AI Executive Summary, and ask questions in the report Q&A chat.

---

## Running Unit Tests

The test suite contains 71 unit tests covering parsing, schema validation, LLM extraction parsing, financial ratio calculations, document chunking, BM25 retrieval, report Q&A grounding, citation filtering, executive summary generation, and upload staleness/hashing:

```bash
python -m unittest discover tests -v
```
