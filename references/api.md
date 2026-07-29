# Croc-Trader REST API Reference

Dokumentation der REST-API-Schnittstellen von Croc-Trader ([app/routes/api.py](file:///Users/produktmanagement/Python/github/croc-trader/app/routes/api.py)).

---

## 🔒 Authentifizierung & Sicherheit

* **IP-Whitelist Schutz**: Endpunkte mit dem Dekorator `@require_ip_whitelist` verlangen Zugriff von zugelassenen IP-Adressen (Konfiguration in `security_config`).
* **Format**: Alle Endpunkte liefern JSON-Antworten mit UTF-8 Kodierung.
* **Content-Type**: `application/json` (für `POST`-Requests wird ein JSON-Body oder URL-Query-Parameter akzeptiert).

---

## 💡 Übersicht wichtiger Steuerungsparameter (Datum & Budget)

| Parameter | Typ | Verfügbar in | Beschreibung & Standardwerte |
| :--- | :--- | :--- | :--- |
| **`budget`** | `float` | Backfill Endpunkte (`/trades/backfill/...`) | **Handelskapital / Positionssumme in USD** pro Trade. Standard: `10000.0` |
| **`start_date`** (oder `start`) | `string` | Backfill Endpunkte (`/trades/backfill/...`) | **Startdatum der Simulation** im Format `YYYY-MM-DD`. Standard: `"2025-01-01"` |
| **`end_date`** (oder `end`) | `string` | Backfill Endpunkte (`/trades/backfill/...`) | **Enddatum der Simulation** im Format `YYYY-MM-DD`. Standard: Aktuelles Tagesdatum |
| **`date`** | `string` | Screener Endpunkte (`/screener/...`) | **Stichtagsdatum** für ein historisches Screening im Format `YYYY-MM-DD`. Standard: Aktueller Handelstag |
| **`days`** | `int` | Screener Endpunkte (`/screener/...`) | **Anzahl Handelstage Lookback** für die Indikatormessung. Standard: `0` |
| **`clear_existing`** (oder `clear`) | `bool` | Backfill Endpunkte (`/trades/backfill/...`) | **Trades zurücksetzen**: Löscht bestehende Backfill-Trades dieser Strategie vor dem Lauf. Standard: `true` |

---

## 1. System & Health Check

### `GET /health`
* **Beschreibung**: Öffentlicher Health-Check Endpunkt zur Überprüfung der Server-Verfügbarkeit.
* **Auth**: Keine (öffentlich).
* **Response `200 OK`**:
  ```json
  {
    "status": "ok"
  }
  ```

### `GET /`
* **Beschreibung**: Authentifizierter System-Root Check.
* **Auth**: IP-Whitelist (`@require_ip_whitelist`).
* **Response `200 OK`**:
  ```json
  {
    "status": "ok"
  }
  ```

---

## 2. Signal Ingestion (Webhooks)

### `POST /webhook`
* **Beschreibung**: Nimmt Handelssignale (z. B. von TradingView) entgegen und speichert sie in `signals.db`.
* **Auth**: IP-Whitelist.
* **Payload (JSON Body)**:
  ```json
  {
    "symbol": "AAPL",
    "strategy": "BridgeScout",
    "timeframe": "1D",
    "signal": "BUY",
    "price": 185.50,
    "timestamp": "2026-07-29T09:00:00Z"
  }
  ```
* **Response `201 Created`**:
  ```json
  {
    "status": "success",
    "id": 142
  }
  ```
* **Response `400 Bad Request`**: Fehlender Parameter (`symbol`) oder invalides JSON.

---

## 3. Screener & Markt-Analyse

### `POST /screener/run`
* **Beschreibung**: Manuelles Anstoßen aller aktiven Handels-Screener.
* **Auth**: IP-Whitelist.
* **Query-Parameter**:
  * `days` *(int, default: `0`)*: Handelstage Lookback.
  * `strategy` *(str, optional)*: Filter für eine bestimmte Strategie.
* **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "stats": { "tgim": 2, "bridge_scout": 1 }
  }
  ```

### `POST /screener/run/<strategy_name>`
* **Beschreibung**: Führt das Screening für eine spezifische Strategie aus.
* **Auth**: IP-Whitelist.
* **Pfad-Parameter**:
  * `strategy_name`: z. B. `tgim`, `bridge-scout`, `bounce-bandit`.
* **Query-Parameter (Datum & Lookback)**:
  * **`date`** *(str, optional, Format: `YYYY-MM-DD`)*: Stichtagsdatum für das Screening.
  * **`days`** *(int, optional, default: `0`)*: Lookback-Handelstage.
* **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "strategy": "bridge_scout",
    "signals_found": 3
  }
  ```

### `POST /screener/dip-buyer`
* **Beschreibung**: Detaillierte Einzelaktien-Analyse für die **DipBuyer** Strategie.
* **Query/JSON Body**: `symbol` *(str, Pflicht)*.

### `POST /screener/turnover`
* **Beschreibung**: Detaillierte Einzelaktien-Analyse für die **TurnoverTiming** Strategie.
* **Query/JSON Body**: `symbol` *(str, Pflicht)*.

### `POST /screener/croc`
* **Beschreibung**: Vollständige Signalliste für die **CrocSetup** Strategie.
* **Query-Parameter**:
  * **`date`** *(str, optional, Format: `YYYY-MM-DD`)*: Analyse-Stichtagsdatum.
  * **`days`** *(int, optional, default: `0`)*: Lookback-Tage.

### `POST /screener/ndx-momentum`
* **Beschreibung**: Marktregime-Bestimmung & Top-Momentum Leader für Nasdaq 100 (**NDX Momentum**).
* **Query-Parameter**:
  * **`date`** *(str, optional, Format: `YYYY-MM-DD`)*: Stichtagsdatum.

---

## 4. Order-Generierung & Execution

### `POST /orders/generate`
* **Beschreibung**: Erzeugt tägliche Bracket Orders (Entry, Stop Loss, Take Profit) und exportiert diese als CSV-Datei nach `data/orders/`.
* **Auth**: IP-Whitelist.
* **Response `201 Created` / `200 OK`**:
  ```json
  {
    "status": "success",
    "file": "data/orders/orders_2026_07_29.csv"
  }
  ```

---

## 5. Backtesting & Historien-Backfill (Datum & Budget)

### `POST /trades/backfill`
* **Beschreibung**: Generischer Backfill-Trigger. Kann über den Query-Parameter `strategy` auch mit Datum und Budget aufgerufen werden.
* **Query-Parameter**:
  * `strategy` *(str, optional)*: Strategiename (`tgim`, `bridge-scout`, `bounce-bandit`).
  * **`start_date`** / **`start`** *(str, default: `"2025-01-01"`)*: Startdatum (YYYY-MM-DD).
  * **`end_date`** / **`end`** *(str, optional, default: heute)*: Enddatum (YYYY-MM-DD).
  * **`budget`** *(float, default: `10000.0`)*: Positionsbudget in USD pro Trade.
  * **`clear_existing`** / **`clear`** *(bool, default: `true`)*: Vorherige Backfill-Trades aus der DB löschen.

### `POST /trades/backfill/<strategy_name>`
* **Beschreibung**: Historische Backfill-Simulation für eine spezifische Strategie über ein beliebiges Datum-Intervall mit anpassbarem Budget.
* **Auth**: IP-Whitelist.
* **Pfad-Parameter**:
  * `strategy_name`: Strategie-Kennung (`tgim`, `bridge-scout`, `bounce-bandit`).
* **Query-Parameter (Datum & Budget)**:
  * **`start_date`** *(oder `start`)*: `string` (Format: `YYYY-MM-DD`, Standard: `"2025-01-01"`). Beispiel: `start_date=2026-01-01`
  * **`end_date`** *(oder `end`)*: `string` (Format: `YYYY-MM-DD`, Standard: aktueller Tag). Beispiel: `end_date=2026-06-30`
  * **`budget`**: `float` (Standard: `10000.0`). Handelsbudget in USD. Beispiel: `budget=5000.0`
  * **`clear_existing`** *(oder `clear`)*: `bool` (Standard: `true`). Beispiel: `clear_existing=true`
* **Aufrufbeispiel**:
  `POST /trades/backfill/bridge-scout?start_date=2026-01-01&end_date=2026-06-30&budget=15000.0`
* **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "result": {
      "start_date": "2026-01-01",
      "end_date": "2026-06-30",
      "signals_generated": 14,
      "trades_filled": 12,
      "trades_closed": 10,
      "total_pnl": 1240.50,
      "win_rate": 80.0,
      "closed_trades": [...]
    }
  }
  ```

---

## 6. Marktdaten-Synchronisation (`stocks.db`)

### `POST /market/sync`
* **Beschreibung**: Startet die inkrementelle oder vollständige Aktualisierung von Kursdaten (yfinance) im Hintergrund.
* **Auth**: IP-Whitelist.
* **Query-Parameter**:
  * `full` *(bool, default: `false`)*: Bei `true` wird ein vollständiger Re-Download durchgeführt.
* **Response `202 Accepted`**:
  ```json
  {
    "status": "accepted",
    "message": "Sync started"
  }
  ```

### `POST /market/reload`
* **Beschreibung**: Triggert einen vollständigen manuellen Re-Download aller Marktdaten im Hintergrund.
* **Auth**: IP-Whitelist.
* **Response `200 OK`**:
  ```json
  {
    "status": "queued",
    "message": "Full reload triggered"
  }
  ```
