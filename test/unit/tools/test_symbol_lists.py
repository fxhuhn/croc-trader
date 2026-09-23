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


def test_refresh_data_fetches_all_indices() -> None:
    """Verifies that _refresh_data invokes _fetch_from_wikipedia for S&P 500, S&P 100, Nasdaq 100, Dow 30, and Russell 1000."""
    es = ExchangeSymbol()
    called_indices = []

    def mock_fetch(
        url: str | list[str], search_columns: list[str], name: str
    ) -> list[str]:
        called_indices.append(name)
        if name == "S&P 100":
            return ["AAPL", "MSFT", "NVDA"]
        if name == "Dow Jones 30":
            return ["AAPL", "MSFT"]
        if name == "Russell 1000":
            return ["AAPL", "MSFT", "NVDA"]
        return ["SPY"]

    with (
        patch.object(es, "_fetch_from_wikipedia", side_effect=mock_fetch),
        patch.object(es, "_save_to_cache"),
    ):
        es._refresh_data()

    assert "S&P 500" in called_indices
    assert "S&P 100" in called_indices
    assert "NASDAQ-100" in called_indices
    assert "Dow Jones 30" in called_indices
    assert "Russell 1000" in called_indices
    assert es.sp_100 == ["AAPL", "MSFT", "NVDA"]
    assert es.dow_30 == ["AAPL", "MSFT"]
    assert es.russell_1000 == ["AAPL", "MSFT", "NVDA"]


def test_sp_100_property_and_all_inclusion() -> None:
    """Verifies that sp_100 property returns a copy and is included in all."""
    es = ExchangeSymbol()
    es._sp_100 = ["AAPL", "GOOGL"]
    es._special_symbols = []
    es._sp_500 = []
    es._nasdaq_100 = []
    es._dow_30 = []
    es._russell_1000 = []

    result = es.sp_100
    assert result == ["AAPL", "GOOGL"]

    # Verify copy isolation
    result.append("NEW_SYM")
    assert es.sp_100 == ["AAPL", "GOOGL"]

    # Verify inclusion in all
    assert "AAPL" in es.all
    assert "GOOGL" in es.all


def test_singleton_returns_same_instance() -> None:
    """Verifies that ExchangeSymbol adheres to the singleton pattern."""
    instance_one = ExchangeSymbol()
    instance_two = ExchangeSymbol()
    assert instance_one is instance_two


def test_load_from_cache_file_not_found(tmp_path: object) -> None:
    """Verifies that missing cache file is handled gracefully without error."""
    import pathlib

    es = ExchangeSymbol()
    non_existent = pathlib.Path(str(tmp_path)) / "missing.json"

    with patch("app.tools.symbol_lists.CACHE_FILE", non_existent):
        # Should not raise
        es._load_from_cache()


def test_load_from_cache_corrupt_file(tmp_path: object) -> None:
    """Verifies that corrupted JSON cache does not crash initialization."""
    import pathlib

    corrupt_file = pathlib.Path(str(tmp_path)) / "corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    es = ExchangeSymbol()
    with patch("app.tools.symbol_lists.CACHE_FILE", corrupt_file):
        es._load_from_cache()


def test_save_and_load_cache_roundtrip(tmp_path: object) -> None:
    """Verifies saving to and loading from cache."""
    import pathlib

    cache_path = pathlib.Path(str(tmp_path)) / "test_cache.json"
    es = ExchangeSymbol()
    es._sp_500 = ["AAPL"]
    es._sp_100 = ["AAPL"]
    es._nasdaq_100 = ["MSFT"]
    es._dow_30 = ["V"]
    es._russell_1000 = ["IWM"]

    with (
        patch("app.tools.symbol_lists.CACHE_FILE", cache_path),
        patch("app.tools.symbol_lists.CACHE_DIR", cache_path.parent),
    ):
        es._save_to_cache()
        assert cache_path.exists()

        # Reset in memory and reload
        es._sp_500 = []
        es._sp_100 = []
        es._load_from_cache()
        assert es.sp_500 == ["AAPL"]
        assert es.sp_100 == ["AAPL"]
        assert es.nasdaq_100 == ["MSFT"]
        assert es.dow_30 == ["V"]
        assert es.russell_1000 == ["IWM"]


def test_fetch_from_wikipedia_no_matching_column() -> None:
    """Verifies that pages without matching columns return empty list."""
    unmatched_df = pd.DataFrame({"Rank": [1, 2], "Company": ["A", "B"]})
    es = ExchangeSymbol()

    with patch("pandas.read_html", return_value=[unmatched_df]):
        result = es._fetch_from_wikipedia(
            url="https://en.wikipedia.org/wiki/Unmatched",
            search_columns=["Ticker", "Symbol"],
            name="Unmatched Index",
        )
    assert result == []


def test_fetch_from_wikipedia_read_error() -> None:
    """Verifies that read_html exceptions are caught and return empty list."""
    es = ExchangeSymbol()

    with patch("pandas.read_html", side_effect=ValueError("Failed to parse HTML")):
        result = es._fetch_from_wikipedia(
            url="https://en.wikipedia.org/wiki/Error",
            search_columns=["Ticker"],
            name="Error Index",
        )
    assert result == []


def test_fetch_from_wikipedia_filters_nan_and_empty() -> None:
    """Verifies that NaN and empty strings are filtered out from symbols."""
    df_with_nans = pd.DataFrame({"Ticker": ["AAPL", "nan", "", "  ", "MSFT"]})
    es = ExchangeSymbol()

    with patch("pandas.read_html", return_value=[df_with_nans]):
        result = es._fetch_from_wikipedia(
            url="https://en.wikipedia.org/wiki/Nans",
            search_columns=["Ticker"],
            name="Nan Index",
        )
    assert result == ["AAPL", "MSFT"]


def test_refresh_data_empty_fetch_does_not_clear_state() -> None:
    """Verifies that an empty fetch from Wikipedia does not clear existing state."""
    es = ExchangeSymbol()
    es._sp_500 = ["AAPL"]
    es._sp_100 = ["AAPL"]

    with (
        patch.object(es, "_fetch_from_wikipedia", return_value=[]),
        patch.object(es, "_save_to_cache"),
    ):
        es._refresh_data()

    assert es.sp_500 == ["AAPL"]
    assert es.sp_100 == ["AAPL"]


def test_refresh_data_exception_handled() -> None:
    """Verifies that exceptions during refresh_data are caught and logged."""
    es = ExchangeSymbol()

    with patch.object(
        es, "_fetch_from_wikipedia", side_effect=RuntimeError("Network down")
    ):
        # Should not raise
        es._refresh_data()


def test_clean_symbol() -> None:
    """Verifies symbol cleaning and rejection rules."""
    from app.tools.symbol_lists import clean_symbol

    assert clean_symbol("AAPL") == "AAPL"
    assert clean_symbol("BRK.B") == "BRK-B"
    assert clean_symbol("  msft  ") == "msft"
    assert clean_symbol("") is None
    assert clean_symbol("   ") is None
    assert clean_symbol("nan") is None
    assert clean_symbol("NaN") is None


def test_extract_symbols_from_tables() -> None:
    """Verifies extracting and deduplicating symbols across tables."""
    from app.tools.symbol_lists import extract_symbols_from_tables

    table_1 = pd.DataFrame({"Rank": [1, 2], "Name": ["Company A", "Company B"]})
    table_2 = pd.DataFrame({"Ticker": ["MSFT", "AAPL", "BRK.B", "nan", ""]})

    symbols = extract_symbols_from_tables([table_1, table_2], ["Symbol", "Ticker"])
    assert symbols == ["AAPL", "BRK-B", "MSFT"]

    # Table with no matching columns
    empty = extract_symbols_from_tables([table_1], ["Symbol", "Ticker"])
    assert empty == []


def test_load_from_cache_non_dict(tmp_path: object) -> None:
    """Verifies that non-dict JSON cache is rejected gracefully."""
    import pathlib

    bad_file = pathlib.Path(str(tmp_path)) / "list_cache.json"
    bad_file.write_text("['not', 'a', 'dict']", encoding="utf-8")

    es = ExchangeSymbol()
    with patch("app.tools.symbol_lists.CACHE_FILE", bad_file):
        es._load_from_cache()


def test_save_to_cache_os_error(tmp_path: object) -> None:
    """Verifies that OSError during cache save is handled cleanly."""
    import pathlib

    test_file = pathlib.Path(str(tmp_path)) / "os_error.json"
    es = ExchangeSymbol()

    with patch("pathlib.Path.open", side_effect=OSError("Mock disk error")):
        with patch("app.tools.symbol_lists.CACHE_FILE", test_file):
            es._save_to_cache()
