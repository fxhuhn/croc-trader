"""Unit tests for formatting utilities."""

from decimal import Decimal

from app.tools.formatting import format_german_number


def test_format_german_number_standard_decimals() -> None:
    """Verifies standard 2-decimal formatting with dot thousands and comma decimal separator."""
    assert format_german_number(1234.56) == "1.234,56"
    assert format_german_number(1000000.5) == "1.000.000,50"
    assert format_german_number(0.0) == "0,00"
    assert format_german_number(42) == "42,00"


def test_format_german_number_custom_decimals() -> None:
    """Verifies formatting with 0, 1, and 3 decimal places."""
    assert format_german_number(100000, decimals=0) == "100.000"
    assert format_german_number(27.64, decimals=1) == "27,6"
    assert format_german_number(27.65, decimals=1) == "27,7"
    assert format_german_number(1234.5678, decimals=3) == "1.234,568"


def test_format_german_number_prefix_plus() -> None:
    """Verifies prefix_plus parameter on positive, negative, and zero values."""
    assert format_german_number(27.6, decimals=1, prefix_plus=True) == "+27,6"
    assert format_german_number(-15.3, decimals=1, prefix_plus=True) == "-15,3"
    assert format_german_number(0.0, decimals=1, prefix_plus=True) == "0,0"


def test_format_german_number_negative_values() -> None:
    """Verifies formatting of negative numbers."""
    assert format_german_number(-1234.56) == "-1.234,56"
    assert format_german_number(-5000, decimals=0) == "-5.000"


def test_format_german_number_zero_normalization() -> None:
    """Verifies that values rounding to zero do not display as -0,0."""
    assert format_german_number(-0.001, decimals=1) == "0,0"
    assert format_german_number(-0.00001, decimals=2) == "0,00"


def test_format_german_number_with_decimal_and_string() -> None:
    """Verifies handling of Decimal and numeric strings."""
    assert format_german_number(Decimal("54321.10")) == "54.321,10"
    assert format_german_number("9876.5", decimals=1) == "9.876,5"


def test_format_german_number_invalid_and_none() -> None:
    """Verifies fallback behavior for None, empty string, and non-numeric inputs."""
    assert format_german_number(None) == "-"
    assert format_german_number(None, default="N/A") == "N/A"
    assert format_german_number("") == "-"
    assert format_german_number("not_a_number") == "not_a_number"
    assert format_german_number(float("nan")) == "-"
    assert format_german_number(float("inf")) == "-"
