import json
import logging

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_tickers_from_github() -> dict[str, set[str]]:
    """
    Lädt die Ticker-Listen für NASDAQ, NYSE und AMEX von GitHub.
    Verwendet die _tickers.json Dateien für einfache Symbol-Listen.
    """
    base_url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"

    exchanges = {
        "nasdaq": f"{base_url}/nasdaq/nasdaq_tickers.json",
        "nyse": f"{base_url}/nyse/nyse_tickers.json",
        "amex": f"{base_url}/amex/amex_tickers.json",
    }

    exchange_lists = {}

    logger.info("Lade Ticker-Listen...")

    for exchange, url in exchanges.items():
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            tickers = json.loads(response.text)
            exchange_lists[exchange] = set(tickers)
            logger.info(f"{exchange.upper()}: {len(tickers)} Symbole geladen")
        except Exception as e:
            logger.warning(f"Fehler beim Laden von {exchange}: {e}")
            return None

    return exchange_lists


def main():
    exchange_lists = load_tickers_from_github()

    if not exchange_lists:
        logger.error("Konnte keine Daten laden.")
        return


if __name__ == "__main__":
   main()
