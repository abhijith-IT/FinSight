"""Validation module for FinSight financial extraction.

Ensures extracted financial figures satisfy accounting consistency,
detects and flags unavailable values, and handles malformed LLM responses.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from extraction.financial_schema import (
    AnnualFinancialData,
    FinancialExtractionReport,
    FinancialValue,
)

CORE_METRIC_PATHS = [
    ("income_statement", "revenue"),
    ("income_statement", "cost_of_goods_sold"),
    ("income_statement", "operating_expenses"),
    ("income_statement", "operating_profit"),
    ("income_statement", "net_profit"),
    ("balance_sheet", "total_assets"),
    ("balance_sheet", "total_liabilities"),
    ("balance_sheet", "total_debt"),
    ("balance_sheet", "equity"),
    ("cash_flow_statement", "operating_cash_flow"),
    ("cash_flow_statement", "capital_expenditure"),
]


class ValidationIssue(BaseModel):
    """Represents a specific finding or warning from validation."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["INFO", "WARNING", "ERROR"] = "WARNING"
    field: str
    message: str
    year: Optional[int] = None


class ValidationReport(BaseModel):
    """Consolidated validation summary for an extraction report."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = True
    issues: List[ValidationIssue] = Field(default_factory=list)
    missing_fields_by_year: Dict[int, List[str]] = Field(default_factory=dict)
    has_errors: bool = False

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]


def clean_and_parse_llm_json(raw_text: str) -> dict:
    """Extract and parse JSON from LLM output, handling code fences or extraneous text.

    Args:
        raw_text: Raw string response from Gemini.

    Returns:
        dict: Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON can be extracted or parsed.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned an empty response.")

    text = raw_text.strip()

    # Strip markdown code blocks if present (```json ... ``` or ``` ...)
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: search for first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON in LLM response: {err}") from err
        raise ValueError(f"Could not parse valid JSON from response: {text[:100]}...")


def sanitize_and_flag_annual_data(annual_data: AnnualFinancialData) -> AnnualFinancialData:
    """Ensure missing values are set to None, flagged as unavailable, and recorded.

    Args:
        annual_data: AnnualFinancialData instance.

    Returns:
        Updated AnnualFinancialData instance with standardized flags.
    """
    missing_list: List[str] = []

    for statement_name, field_name in CORE_METRIC_PATHS:
        statement = getattr(annual_data, statement_name, None)
        if statement is None:
            continue

        metric: Optional[FinancialValue] = getattr(statement, field_name, None)
        if metric is None or metric.value is None:
            setattr(
                statement,
                field_name,
                FinancialValue(value=None, source_page=None, is_unavailable=True),
            )
            missing_list.append(field_name)
        else:
            metric.is_unavailable = False

    annual_data.missing_fields = sorted(list(set(missing_list)))
    annual_data.source_pages = sorted(list(set(annual_data.source_pages)))
    return annual_data


def validate_financial_report(report: FinancialExtractionReport) -> ValidationReport:
    """Perform consistency and sanity validations on extracted financial data.

    Checks:
    - Normalizes and flags unavailable / null fields.
    - Validates reporting years (1900-2100).
    - Checks the Balance Sheet equation: Total Assets == Total Liabilities + Equity.
    - Flags unusual negative values where typically positive (e.g., negative revenue).
    - Checks Net Profit relative to Revenue.

    Args:
        report: The extracted financial report.

    Returns:
        ValidationReport: Summary of all validation findings.
    """
    issues: List[ValidationIssue] = []
    missing_by_year: Dict[int, List[str]] = {}
    has_errors = False

    for annual_data in report.annual_reports:
        # Standardize missing flags and nulls
        sanitize_and_flag_annual_data(annual_data)
        missing_by_year[annual_data.year] = annual_data.missing_fields

        year = annual_data.year

        # 1. Year Range Sanity Check
        if year < 1900 or year > 2100:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    field="year",
                    message=f"Extracted reporting year {year} is out of realistic range (1900-2100).",
                    year=year,
                )
            )
            has_errors = True

        # 2. Accounting Equation Check: Assets = Liabilities + Equity
        assets = annual_data.balance_sheet.total_assets.value
        liabilities = annual_data.balance_sheet.total_liabilities.value
        equity = annual_data.balance_sheet.equity.value

        if assets is not None and liabilities is not None and equity is not None:
            expected_assets = liabilities + equity
            discrepancy = abs(assets - expected_assets)
            scale = max(abs(assets), 1.0)
            relative_error = discrepancy / scale

            # Allow small rounding threshold (2%)
            if relative_error > 0.02:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        field="balance_sheet",
                        message=(
                            f"Balance Sheet equation discrepancy for {year}: "
                            f"Total Assets ({assets}) != Liabilities ({liabilities}) + Equity ({equity}). "
                            f"Discrepancy: {discrepancy:.2f} ({relative_error * 100:.1f}%)."
                        ),
                        year=year,
                    )
                )

        # 3. Revenue Check
        rev = annual_data.income_statement.revenue.value
        if rev is not None and rev < 0:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    field="revenue",
                    message=f"Negative revenue ({rev}) reported for year {year}.",
                    year=year,
                )
            )

        # 4. Net Profit vs Revenue Sanity Check
        net_profit = annual_data.income_statement.net_profit.value
        if rev is not None and net_profit is not None and rev > 0:
            if net_profit > (rev * 1.5):  # Net profit substantially greater than revenue
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        field="net_profit",
                        message=(
                            f"Net profit ({net_profit}) substantially exceeds revenue ({rev}) for year {year}."
                        ),
                        year=year,
                    )
                )

        # 5. Missing Fields Notification
        for missing_field in annual_data.missing_fields:
            issues.append(
                ValidationIssue(
                    severity="INFO",
                    field=missing_field,
                    message=f"Metric '{missing_field}' was not reported or unavailable for year {year}.",
                    year=year,
                )
            )

    return ValidationReport(
        is_valid=not has_errors,
        issues=issues,
        missing_fields_by_year=missing_by_year,
        has_errors=has_errors,
    )
