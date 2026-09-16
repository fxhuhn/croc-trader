"""Formatting utilities for localization and display presentation."""

import math
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


def format_german_number(
    value: float | int | Decimal | str | None,
    decimals: int = 2,
    prefix_plus: bool = False,
    default: str = "-",
) -> str:
    """Formats a numeric value using German locale conventions (e.g. 1.234,56).

    Dot is used as the thousands separator and comma as the decimal separator.
    Commercial rounding (ROUND_HALF_UP) is used to avoid IEEE 754 floating point artifacts.

    Args:
        value: Numeric value (float, int, Decimal, or numeric str), or None.
        decimals: Number of decimal places (default: 2).
        prefix_plus: If True, prepends '+' for strictly positive numbers.
        default: Fallback string when value is None or not a valid number.

    Returns:
        Formatted string in German format (e.g. '1.xxx,xx').
    """
    if value is None or value == "":
        return default

    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return default
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return str(value)

    if decimal_value.is_nan() or decimal_value.is_infinite():
        return default

    quantize_exponent = Decimal("10") ** -decimals if decimals > 0 else Decimal("1")
    rounded_decimal = decimal_value.quantize(quantize_exponent, rounding=ROUND_HALF_UP)

    # Normalize values that round to 0 to avoid '-0,0'
    if rounded_decimal == 0:
        rounded_decimal = Decimal("0")

    formatted = f"{abs(rounded_decimal):,.{decimals}f}"
    german_formatted = formatted.translate(str.maketrans(",.", ".,"))

    prefix = ""
    if rounded_decimal < 0:
        prefix = "-"
    elif prefix_plus and rounded_decimal > 0:
        prefix = "+"

    return f"{prefix}{german_formatted}"
