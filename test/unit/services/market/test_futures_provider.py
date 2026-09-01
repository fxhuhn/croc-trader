"""Tests for futures contract resolution and front-month rollover heuristic.

Tests the pure functions in futures_provider.py against known quarterly
expiry dates and edge cases around the 3rd Friday rollover boundary.
"""

from datetime import date

import pytest

from app.services.market.futures_provider import (
    FuturesContract,
    _advance_quarter,
    _current_or_next_quarterly_month,
    _find_next_expiry,
    _find_third_friday,
    resolve_front_month_contract,
)


class TestFindThirdFriday:
    """Tests for the 3rd Friday calculation."""

    def test_september_2026(self) -> None:
        """Sep 2026: 3rd Friday is 2026-09-18."""
        assert _find_third_friday(2026, 9) == date(2026, 9, 18)

    def test_december_2026(self) -> None:
        """Dec 2026: 3rd Friday is 2026-12-18."""
        assert _find_third_friday(2026, 12) == date(2026, 12, 18)

    def test_march_2027(self) -> None:
        """Mar 2027: 3rd Friday is 2027-03-19."""
        assert _find_third_friday(2027, 3) == date(2027, 3, 19)

    def test_june_2026(self) -> None:
        """Jun 2026: 3rd Friday is 2026-06-19."""
        assert _find_third_friday(2026, 6) == date(2026, 6, 19)


class TestCurrentOrNextQuarterlyMonth:
    """Tests for quarterly month resolution."""

    @pytest.mark.parametrize(
        ("input_month", "expected"),
        [
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 6),
            (5, 6),
            (6, 6),
            (7, 9),
            (8, 9),
            (9, 9),
            (10, 12),
            (11, 12),
            (12, 12),
        ],
    )
    def test_all_months(self, input_month: int, expected: int) -> None:
        assert _current_or_next_quarterly_month(input_month) == expected


class TestAdvanceQuarter:
    """Tests for quarter advancement with year wrapping."""

    def test_march_to_june(self) -> None:
        assert _advance_quarter(3, 2026) == (6, 2026)

    def test_june_to_september(self) -> None:
        assert _advance_quarter(6, 2026) == (9, 2026)

    def test_september_to_december(self) -> None:
        assert _advance_quarter(9, 2026) == (12, 2026)

    def test_december_wraps_to_next_year(self) -> None:
        assert _advance_quarter(12, 2026) == (3, 2027)


class TestFindNextExpiry:
    """Tests for expiry resolution around the 3rd Friday boundary."""

    def test_before_third_friday_stays_in_quarter(self) -> None:
        """August 31 is before Sep 18 (3rd Friday) → Sep contract."""
        assert _find_next_expiry(date(2026, 8, 31)) == (9, 2026)

    def test_on_third_friday_rolls_forward(self) -> None:
        """On Sep 18 (3rd Friday) → rolls to Dec contract."""
        assert _find_next_expiry(date(2026, 9, 18)) == (12, 2026)

    def test_after_third_friday_rolls_forward(self) -> None:
        """Sep 19 is after 3rd Friday → Dec contract."""
        assert _find_next_expiry(date(2026, 9, 19)) == (12, 2026)

    def test_mid_quarter_month(self) -> None:
        """July 15 is well before Sep 18 → Sep contract."""
        assert _find_next_expiry(date(2026, 7, 15)) == (9, 2026)

    def test_december_rollover_wraps_year(self) -> None:
        """After Dec 3rd Friday → Mar of next year."""
        assert _find_next_expiry(date(2026, 12, 19)) == (3, 2027)

    def test_early_january(self) -> None:
        """Jan 5 → Mar contract (next quarterly month)."""
        assert _find_next_expiry(date(2027, 1, 5)) == (3, 2027)


class TestResolveFrontMonthContract:
    """Tests for the full contract resolution pipeline."""

    def test_mnq_september_2026(self) -> None:
        """MNQ before Sep 3rd Friday → MNQU2026."""
        contract = resolve_front_month_contract("MNQ", date(2026, 8, 31))
        assert contract == FuturesContract(
            base_symbol="MNQ",
            tv_symbol="MNQU2026",
            exchange="CME_MINI",
            month_code="U",
            expiry_month=9,
            expiry_year=2026,
        )

    def test_mes_september_2026(self) -> None:
        """MES before Sep 3rd Friday → MESU2026."""
        contract = resolve_front_month_contract("MES", date(2026, 8, 31))
        assert contract == FuturesContract(
            base_symbol="MES",
            tv_symbol="MESU2026",
            exchange="CME_MINI",
            month_code="U",
            expiry_month=9,
            expiry_year=2026,
        )

    def test_mnq_after_september_rollover(self) -> None:
        """MNQ after Sep 3rd Friday → MNQZ2026."""
        contract = resolve_front_month_contract("MNQ", date(2026, 9, 20))
        assert contract.tv_symbol == "MNQZ2026"
        assert contract.month_code == "Z"
        assert contract.expiry_month == 12

    def test_mnq_december_rollover_to_next_year(self) -> None:
        """MNQ after Dec 3rd Friday → MNQH2027."""
        contract = resolve_front_month_contract("MNQ", date(2026, 12, 19))
        assert contract.tv_symbol == "MNQH2027"
        assert contract.expiry_year == 2027
        assert contract.month_code == "H"

    def test_unknown_symbol_raises(self) -> None:
        """Unknown symbol raises ValueError."""
        with pytest.raises(ValueError, match="Unknown futures symbol"):
            resolve_front_month_contract("INVALID", date(2026, 8, 31))
