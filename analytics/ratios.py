"""Financial Ratio Engine for FinSight.

Provides pure Python financial ratio calculations (Margins, ROE, Debt-to-Equity,
and Multi-Year YoY Revenue Growth) directly from extracted AnnualFinancialData models.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from extraction.financial_schema import (
    AnnualFinancialData,
    FinancialExtractionReport,
)


class FinancialRatios(BaseModel):
    """Calculated financial ratios for a given reporting year."""

    model_config = ConfigDict(extra="ignore")

    year: int = Field(..., description="Reporting fiscal year")
    gross_margin_pct: Optional[float] = Field(
        default=None,
        description="Gross Margin % = (Revenue - COGS) / Revenue * 100",
    )
    net_profit_margin_pct: Optional[float] = Field(
        default=None,
        description="Net Profit Margin % = Net Profit / Revenue * 100",
    )
    roe: Optional[float] = Field(
        default=None,
        description="Return on Equity % = Net Profit / Shareholders' Equity * 100",
    )
    debt_to_equity: Optional[float] = Field(
        default=None,
        description="Debt-to-Equity Ratio = Total Debt / Shareholders' Equity",
    )
    revenue_growth_yoy_pct: Optional[float] = Field(
        default=None,
        description="YoY Revenue Growth % = (Current Revenue - Previous Revenue) / Previous Revenue * 100",
    )

    @property
    def roe_pct(self) -> Optional[float]:
        """Convenience alias for ROE percentage."""
        return self.roe


def _extract_number(val: Any) -> Optional[float]:
    """Safely extract a float number from a scalar, FinancialValue, or None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, "value"):
        v = getattr(val, "value")
        return float(v) if v is not None else None
    return None


def _format_result(val: float, round_digits: Optional[int] = 2) -> float:
    """Format float result with optional rounding."""
    if round_digits is not None:
        return round(val, round_digits)
    return val


def calculate_gross_margin_pct(
    revenue: Optional[float],
    cost_of_goods_sold: Optional[float],
    round_digits: Optional[int] = 2,
) -> Optional[float]:
    """Calculate Gross Margin % = (Revenue - Cost of Goods Sold) / Revenue * 100.

    Returns None if inputs are missing or revenue is zero.
    """
    rev = _extract_number(revenue)
    cogs = _extract_number(cost_of_goods_sold)

    if rev is None or cogs is None:
        return None
    if rev == 0.0:
        return None

    result = ((rev - cogs) / rev) * 100.0
    return _format_result(result, round_digits)


def calculate_net_profit_margin_pct(
    net_profit: Optional[float],
    revenue: Optional[float],
    round_digits: Optional[int] = 2,
) -> Optional[float]:
    """Calculate Net Profit Margin % = Net Profit / Revenue * 100.

    Returns None if inputs are missing or revenue is zero.
    """
    np = _extract_number(net_profit)
    rev = _extract_number(revenue)

    if np is None or rev is None:
        return None
    if rev == 0.0:
        return None

    result = (np / rev) * 100.0
    return _format_result(result, round_digits)


def calculate_roe(
    net_profit: Optional[float],
    equity: Optional[float],
    round_digits: Optional[int] = 2,
) -> Optional[float]:
    """Calculate ROE = Net Profit / Shareholders' Equity * 100.

    Returns None if inputs are missing or equity is zero.
    """
    np = _extract_number(net_profit)
    eq = _extract_number(equity)

    if np is None or eq is None:
        return None
    if eq == 0.0:
        return None

    result = (np / eq) * 100.0
    return _format_result(result, round_digits)


def calculate_debt_to_equity(
    total_debt: Optional[float],
    equity: Optional[float],
    round_digits: Optional[int] = 2,
) -> Optional[float]:
    """Calculate Debt-to-Equity = Total Debt / Shareholders' Equity.

    NOTE: Total liabilities are NOT treated as total debt.
    Returns None if inputs are missing or equity is zero.
    """
    debt = _extract_number(total_debt)
    eq = _extract_number(equity)

    if debt is None or eq is None:
        return None
    if eq == 0.0:
        return None

    result = debt / eq
    return _format_result(result, round_digits)


def calculate_revenue_growth_yoy_pct(
    current_revenue: Optional[float],
    previous_revenue: Optional[float],
    round_digits: Optional[int] = 2,
) -> Optional[float]:
    """Calculate YoY Revenue Growth % = (Current Revenue - Previous Revenue) / Previous Revenue * 100.

    Returns None if inputs are missing or previous_revenue is zero.
    """
    curr = _extract_number(current_revenue)
    prev = _extract_number(previous_revenue)

    if curr is None or prev is None:
        return None
    if prev == 0.0:
        return None

    result = ((curr - prev) / prev) * 100.0
    return _format_result(result, round_digits)


def calculate_annual_ratios(
    current: AnnualFinancialData,
    previous: Optional[AnnualFinancialData] = None,
    *,
    cost_of_goods_sold: Optional[float] = None,
    total_debt: Optional[float] = None,
    round_digits: Optional[int] = 2,
    strict_consecutive: bool = True,
) -> FinancialRatios:
    """Calculate financial ratios for a single AnnualFinancialData record.

    Args:
        current: Financial data for the target year.
        previous: Optional financial data for the preceding year for YoY growth.
        cost_of_goods_sold: Optional COGS value if not in schema.
        total_debt: Optional Total Debt value (total_liabilities is NOT treated as debt).
        round_digits: Decimal places to round to, or None for exact float.
        strict_consecutive: If True, YoY requires previous.year == current.year - 1.

    Returns:
        FinancialRatios model populated with calculated metrics or None for missing data.
    """
    # 1. Revenue & Net Profit from Income Statement
    revenue = _extract_number(getattr(current.income_statement, "revenue", None))
    net_profit = _extract_number(getattr(current.income_statement, "net_profit", None))

    # Check for COGS on current.income_statement or passed directly (never inferred)
    cogs = cost_of_goods_sold
    if cogs is None:
        cogs = _extract_number(getattr(current.income_statement, "cost_of_goods_sold", None))

    gross_margin = calculate_gross_margin_pct(revenue, cogs, round_digits=round_digits)
    net_margin = calculate_net_profit_margin_pct(net_profit, revenue, round_digits=round_digits)

    # 2. Equity from Balance Sheet
    equity = _extract_number(getattr(current.balance_sheet, "equity", None))
    roe_val = calculate_roe(net_profit, equity, round_digits=round_digits)

    # 3. Debt-to-Equity (strictly requires explicit debt, never total_liabilities)
    debt = total_debt
    if debt is None:
        debt = _extract_number(getattr(current.balance_sheet, "total_debt", None))

    debt_to_equity = calculate_debt_to_equity(debt, equity, round_digits=round_digits)

    # 4. YoY Revenue Growth
    yoy_growth: Optional[float] = None
    if previous is not None:
        is_consecutive = (current.year - previous.year == 1) if strict_consecutive else True
        if is_consecutive:
            prev_revenue = _extract_number(getattr(previous.income_statement, "revenue", None))
            yoy_growth = calculate_revenue_growth_yoy_pct(
                revenue, prev_revenue, round_digits=round_digits
            )

    return FinancialRatios(
        year=current.year,
        gross_margin_pct=gross_margin,
        net_profit_margin_pct=net_margin,
        roe=roe_val,
        debt_to_equity=debt_to_equity,
        revenue_growth_yoy_pct=yoy_growth,
    )


def calculate_ratios_series(
    annual_reports: List[AnnualFinancialData],
    *,
    round_digits: Optional[int] = 2,
    strict_consecutive: bool = True,
) -> List[FinancialRatios]:
    """Calculate financial ratios across a series of AnnualFinancialData records.

    Sorts chronologically by year and calculates YoY growth for consecutive available years.

    Args:
        annual_reports: List of AnnualFinancialData records.
        round_digits: Decimal places to round to, or None for exact float.
        strict_consecutive: If True, YoY requires previous.year == current.year - 1.

    Returns:
        List of FinancialRatios sorted chronologically by year.
    """
    if not annual_reports:
        return []

    # Sort reports chronologically by fiscal year
    sorted_reports = sorted(annual_reports, key=lambda a: a.year)

    # Create mapping of year -> report
    year_map = {report.year: report for report in sorted_reports}

    results: List[FinancialRatios] = []
    for report in sorted_reports:
        # Check if the consecutive previous year exists in available reports
        prev_report: Optional[AnnualFinancialData] = None
        if strict_consecutive:
            prev_report = year_map.get(report.year - 1)
        else:
            # Index of immediately preceding available year
            idx = sorted_reports.index(report)
            if idx > 0:
                prev_report = sorted_reports[idx - 1]

        ratios = calculate_annual_ratios(
            current=report,
            previous=prev_report,
            round_digits=round_digits,
            strict_consecutive=strict_consecutive,
        )
        results.append(ratios)

    return results


def calculate_report_ratios(
    report: FinancialExtractionReport,
    *,
    round_digits: Optional[int] = 2,
    strict_consecutive: bool = True,
) -> List[FinancialRatios]:
    """Calculate financial ratios for all annual reports in a FinancialExtractionReport.

    Args:
        report: FinancialExtractionReport containing annual_reports.
        round_digits: Decimal precision.
        strict_consecutive: Require year difference of 1 for YoY.

    Returns:
        List of FinancialRatios for each year.
    """
    return calculate_ratios_series(
        report.annual_reports,
        round_digits=round_digits,
        strict_consecutive=strict_consecutive,
    )
