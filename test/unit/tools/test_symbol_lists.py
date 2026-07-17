from unittest.mock import patch

import pandas as pd

from app.tools.symbol_lists import ExchangeSymbol


def test_fetch_from_wikipedia_single_url_success() -> None:
    """Verifies fetching symbols from a single Wikipedia URL when table matches."""
    fake_df = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "GOOGL"]})
    es = ExchangeSymbol()

    with patch("pandas.read_html", return_value=[fake_df]):
        result = es._fetch_from_wikipedia(
            url="https://en.wikipedia.org/wiki/Test",
            search_columns=["Ticker", "Symbol"],
            name="Test Index",
        )

    assert result == ["AAPL", "GOOGL", "MSFT"]


def test_fetch_from_wikipedia_fallback_url() -> None:
    """Verifies that fallback URLs are tried if the primary URL yields no matching table."""
    fake_df_fail = pd.DataFrame({"Category": ["A", "B"]})
    fake_df_pass = pd.DataFrame({"Symbol": ["AMZN", "NVDA"]})
    es = ExchangeSymbol()

    def mock_read_html(url: str, **kwargs: object) -> list[pd.DataFrame]:
        if "primary" in url:
            return [fake_df_fail]
        return [fake_df_pass]

    with patch("pandas.read_html", side_effect=mock_read_html):
        result = es._fetch_from_wikipedia(
            url=[
                "https://en.wikipedia.org/wiki/primary",
                "https://en.wikipedia.org/wiki/fallback",
            ],
            search_columns=["Ticker", "Symbol"],
            name="Test Index",
        )

    assert result == ["AMZN", "NVDA"]


def test_fetch_from_wikipedia_dot_cleaning() -> None:
    """Verifies that ticker dot symbols are converted to hyphens (e.g. BRK.B -> BRK-B)."""
    fake_df = pd.DataFrame({"Ticker": ["BRK.B", "BF.B", "AAPL"]})
    es = ExchangeSymbol()

    with patch("pandas.read_html", return_value=[fake_df]):
        result = es._fetch_from_wikipedia(
            url="https://en.wikipedia.org/wiki/Test",
            search_columns=["Ticker"],
            name="Test Index",
        )

    assert result == ["AAPL", "BF-B", "BRK-B"]
