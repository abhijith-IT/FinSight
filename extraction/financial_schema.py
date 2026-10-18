"""Pydantic schemas for structured financial data extraction in FinSight."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FinancialValue(BaseModel):
    """Represents an individual extracted financial metric with value and audit provenance."""

    model_config = ConfigDict(extra="ignore")

    value: Optional[float] = Field(
        default=None,
        description="Numeric value as reported in the financial statements. Null if not reported.",
    )
    source_page: Optional[int] = Field(
        default=None,
        description="PDF page number (1-indexed) where this specific figure was reported.",
    )
    is_unavailable: bool = Field(
        default=False,
        description="Flag set to True if this metric was not reported or unavailable in the source text.",
    )


class IncomeStatement(BaseModel):
    """Extracted Income Statement metrics."""

    model_config = ConfigDict(extra="ignore")

    revenue: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Total revenue, turnover, or net sales.",
    )
    cost_of_goods_sold: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Cost of goods sold (COGS), cost of sales, or cost of revenue.",
    )
    operating_expenses: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Total operating expenses (OPEX).",
    )
    operating_profit: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Operating income, operating profit, or EBIT.",
    )
    net_profit: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Net income, net earnings, or profit after tax.",
    )

    @field_validator(
        "revenue",
        "cost_of_goods_sold",
        "operating_expenses",
        "operating_profit",
        "net_profit",
        mode="before",
    )
    @classmethod
    def _coerce_financial_value(cls, v: Any) -> Any:
        if v is None:
            return FinancialValue(is_unavailable=True)
        return v


class BalanceSheet(BaseModel):
    """Extracted Balance Sheet metrics."""

    model_config = ConfigDict(extra="ignore")

    total_assets: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Total assets of the reporting entity.",
    )
    total_liabilities: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Total liabilities of the reporting entity.",
    )
    total_debt: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Total debt (short-term borrowings + long-term debt). Distinct from total liabilities.",
    )
    equity: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Total shareholders' equity, stockholders' equity, or net worth.",
    )

    @field_validator(
        "total_assets",
        "total_liabilities",
        "total_debt",
        "equity",
        mode="before",
    )
    @classmethod
    def _coerce_financial_value(cls, v: Any) -> Any:
        if v is None:
            return FinancialValue(is_unavailable=True)
        return v


class CashFlowStatement(BaseModel):
    """Extracted Cash Flow Statement metrics."""

    model_config = ConfigDict(extra="ignore")

    operating_cash_flow: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Net cash flow provided by (or used in) operating activities.",
    )
    capital_expenditure: Optional[FinancialValue] = Field(
        default_factory=FinancialValue,
        description="Capital expenditures (CapEx or additions to property, plant, and equipment).",
    )

    @field_validator(
        "operating_cash_flow",
        "capital_expenditure",
        mode="before",
    )
    @classmethod
    def _coerce_financial_value(cls, v: Any) -> Any:
        if v is None:
            return FinancialValue(is_unavailable=True)
        return v


class AnnualFinancialData(BaseModel):
    """Complete financial statements and metadata for a single reporting year."""

    model_config = ConfigDict(extra="ignore")

    year: int = Field(
        ...,
        description="Reporting fiscal year (e.g., 2022, 2023, 2024).",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code or symbol (e.g., USD, EUR, INR, GBP).",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of scale (e.g., millions, billions, thousands, units).",
    )
    income_statement: IncomeStatement = Field(
        default_factory=IncomeStatement,
        description="Extracted Income Statement items.",
    )
    balance_sheet: BalanceSheet = Field(
        default_factory=BalanceSheet,
        description="Extracted Balance Sheet items.",
    )
    cash_flow_statement: CashFlowStatement = Field(
        default_factory=CashFlowStatement,
        description="Extracted Cash Flow Statement items.",
    )
    source_pages: List[int] = Field(
        default_factory=list,
        description="All page numbers where data for this fiscal year was found.",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Explicit list of standard metric names that were not reported for this year.",
    )


class FinancialExtractionReport(BaseModel):
    """Top-level container for all extracted financial data from a document."""

    model_config = ConfigDict(extra="ignore")

    company_name: Optional[str] = Field(
        default=None,
        description="Name of the company or reporting entity if identified.",
    )
    annual_reports: List[AnnualFinancialData] = Field(
        default_factory=list,
        description="Financial data extracted for each reporting year.",
    )
    extraction_notes: List[str] = Field(
        default_factory=list,
        description="Audit notes, section metadata, or extraction warnings.",
    )
