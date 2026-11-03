"""FinSight Generation package.

Provides automated AI narrative generation, including grounded executive summaries.
"""

from generation.export import generate_consolidated_csv, generate_executive_pdf
from generation.summary import ExecutiveSummary, ExecutiveSummaryGenerator

__all__ = [
    "ExecutiveSummary",
    "ExecutiveSummaryGenerator",
    "generate_consolidated_csv",
    "generate_executive_pdf",
]
