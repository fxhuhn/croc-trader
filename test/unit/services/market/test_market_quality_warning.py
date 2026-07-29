"""Unit tests for MarketQualityService completeness warnings."""

from unittest.mock import MagicMock

from app.services.market.quality import MarketQualityService
from app.types import TradeStatus


def test_check_last_trading_day_completeness_all_up_to_date() -> None:
    """Validates that completeness check returns True when all symbols are up to date."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_trade_repo = MagicMock()

    mock_updater.repo = mock_repo
    mock_updater.trade_repository = mock_trade_repo

    mock_repo.get_all_known_symbols.return_value = ["QQQ", "SPY", "AAPL", "MSFT"]
    mock_repo.get_outdated_symbols.return_value = []
    mock_trade_repo.get_by_status.return_value = [
        {"symbol": "AAPL", "status": TradeStatus.ACTIVE.value}
    ]

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    result = service.check_last_trading_day_completeness()

    assert result is True
    mock_telegram.send_message.assert_not_called()


def test_check_last_trading_day_completeness_critical_symbol_missing() -> None:
    """Validates that warning is triggered when a critical symbol (QQQ) is outdated."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_trade_repo = MagicMock()

    mock_updater.repo = mock_repo
    mock_updater.trade_repository = mock_trade_repo

    mock_repo.get_all_known_symbols.return_value = ["QQQ", "SPY", "AAPL", "MSFT"]
    mock_repo.get_outdated_symbols.return_value = ["QQQ"]
    mock_trade_repo.get_by_status.return_value = []

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    result = service.check_last_trading_day_completeness()

    assert result is False
    mock_telegram.send_message.assert_called_once()
    sent_text = mock_telegram.send_message.call_args[0][0]
    assert "WARNUNG: Marktdaten" in sent_text
    assert "QQQ" in sent_text


def test_check_last_trading_day_completeness_active_trade_symbol_missing() -> None:
    """Validates warning when an active trade symbol (NVDA) is missing."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_trade_repo = MagicMock()

    mock_updater.repo = mock_repo
    mock_updater.trade_repository = mock_trade_repo

    mock_repo.get_all_known_symbols.return_value = ["QQQ", "SPY", "NVDA", "MSFT"]
    mock_repo.get_outdated_symbols.return_value = ["NVDA"]
    mock_trade_repo.get_by_status.return_value = [
        {"symbol": "NVDA", "status": TradeStatus.CREATED.value}
    ]

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    result = service.check_last_trading_day_completeness()

    assert result is False
    mock_telegram.send_message.assert_called_once()
    sent_text = mock_telegram.send_message.call_args[0][0]
    assert "NVDA" in sent_text


def test_check_last_trading_day_completeness_missing_ratio_exceeded() -> None:
    """Validates warning when missing ratio exceeds allowed threshold."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_trade_repo = MagicMock()

    mock_updater.repo = mock_repo
    mock_updater.trade_repository = mock_trade_repo

    all_syms = [f"SYM_{i}" for i in range(100)]
    outdated_syms = [f"SYM_{i}" for i in range(10)]  # 10% missing

    mock_repo.get_all_known_symbols.return_value = all_syms
    mock_repo.get_outdated_symbols.return_value = outdated_syms
    mock_trade_repo.get_by_status.return_value = []

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    result = service.check_last_trading_day_completeness(max_allowed_missing_ratio=0.05)

    assert result is False
    mock_telegram.send_message.assert_called_once()
