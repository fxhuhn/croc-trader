# Croc-Trader REST API Reference

Dokumentation der REST-API-Schnittstellen von Croc-Trader (`app/routes/api.py`).

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
| **`start_date`** *(Alias `start`)* | `string` | Backfill Endpunkte (`/trades/backfill/...`) | **Startdatum der Simulation** im Format `YYYY-MM-DD`. Standard: `"2025-01-01"` |
| **`end_date`** *(Alias `end`)* | `string` | Backfill Endpunkte (`/trades/backfill/...`) | **Enddatum der Simulation** im Format `YYYY-MM-DD`. Standard: Aktuelles Tagesdatum |
| **`date`** | `string` | Screener Endpunkte (`/screener/...`) | **Stichtagsdatum** für ein historisches Screening im Format `YYYY-MM-DD`. Standard: Aktueller Handelstag |
| **`days`** | `int` | Screener Endpunkte (`/screener/...`) | **Anzahl Handelstage Lookback** für die Indikatormessung. Standard: `0` |
| **`clear_existing`** *(Alias `clear`)* | `bool` | Backfill Endpunkte (`/trades/backfill/...`) | **Trades zurücksetzen**: Löscht bestehende Backfill-Trades dieser Strategie vor dem Lauf. Standard: `true` |
| **`provider`** | `string` | Marktdaten Endpunkte (`/market/...`) | **Datenprovider-Modus**: `"auto"`, `"yahoo"`, `"tv"`. Standard: `"auto"` |
| **`ignore_today`** | `bool` | Marktdaten Endpunkte (`/market/...`) | **Heutige Intraday-Bars ignorieren**: Standard: `false` |

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

## 3. EOD Pipeline Orchestration

### `POST /pipeline/run`
* **Beschreibung**: Führt die synchrone End-of-Day (EOD) Pipeline sequenziell aus: TradeManager & Position Updates -> Screener Engine -> Order Generierung -> Cache Pre-warming.
* **Auth**: IP-Whitelist.
* **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "timestamp": "2026-08-28T08:20:00",
    "duration_seconds": 12.4,
    "steps_completed": ["trade_manager", "screener", "orders", "cache"]
  }
  ```
* **Response `500 Internal Server Error`**: Bei unerwartetem Pipeline-Fehler.

---

## 4. Screener & Markt-Analyse

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
    "stats": { "tgim": 2, "bridge_scout": 1, "dip_buyer": 0 }
  }
  ```
* **Response `503 Service Unavailable`**: Wenn Screener Engine nicht initialisiert ist.

### `POST /screener/run/<strategy_name>`
* **Beschreibung**: Führt das Screening für eine spezifische Strategie aus. Unterstützt kanonische Strategiebezeichner sowie alle registrierten Aliase (z. B. `qqq_meanrev`, `bridge-scout`, `thank_god_its_monday`, `turnover timing`, `croc_hold`, `croc_split`, `twopercentstrategy`, `dipbuyer`).
* **Auth**: IP-Whitelist.
* **Pfad-Parameter**:
  * `strategy_name`: Strategiekennung oder Alias (z. B. `tgim`, `bridge-scout`, `bounce-bandit`, `dip_buyer`).
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
* **Response `404 Not Found`**: Wenn die Strategie nicht existiert oder nicht in der Engine initialisiert ist:
  ```json
  {
    "status": "error",
    "message": "Strategy 'unknown_strat' not found"
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

### `POST /screener/<strategy_name>/debug`
* **Beschreibung**: Universeller Inspektions- und Debug-Endpunkt für alle registrierten Screener-Strategien. Führt dynamisch Symbol-Analysen (wenn `symbol` übergeben wurde) oder Signal-/Regime-Berechnungen (`days`, `date`) aus.
* **Auth**: IP-Whitelist.
* **Pfad-Parameter**:
  * `strategy_name`: Strategiebezeichner oder Alias (z. B. `dip_buyer`, `tgim`, `bridge-scout`, `croc_setup`).
* **Query / JSON Body**:
  * `symbol` *(str, optional)*: Aktien-Ticker für Einzelaktienprüfung.
  * `days` *(int, optional)*: Lookback-Tage.
  * `date` *(str, optional)*: Stichtagsdatum (YYYY-MM-DD).
* **Response `200 OK`**: JSON-Inspektionsdaten der jeweiligen Strategie.
* **Response `404 Not Found`**: Wenn Strategie nicht existiert oder nicht in der Engine initialisiert ist.

---

## 5. Order-Generierung & Execution

### `POST /orders/generate`
* **Beschreibung**: Erzeugt tägliche Bracket Orders (Entry, Stop Loss, Take Profit) und exportiert diese atomar als CSV-Datei nach `data/orders/`.
* **Auth**: IP-Whitelist.
* **Response `201 Created`**: Wenn neue Orders exportiert wurden.
  ```json
  {
    "status": "success",
    "file": "data/orders/orders_2026_07_29.csv"
  }
  ```
* **Response `200 OK`**: Wenn für den aktuellen Handelstag keine Orders anstanden.
  ```json
  {
    "status": "success",
    "message": "No orders generated"
  }
  ```

---

## 6. Backtesting & Historien-Backfill (Datum & Budget)

### `POST /trades/backfill`
* **Beschreibung**: Generischer Backfill-Trigger. Führt bei fehlendem `strategy`-Parameter den täglichen TradeManager-Prozess aus oder delegiert bei vorhandenem Parameter an die Strategie-Backfill-Simulation.
* **Auth**: IP-Whitelist.
* **Query-Parameter**:
  * `strategy` *(str, optional)*: Strategiename (`tgim`, `bridge-scout`, `bounce-bandit`).
  * **`start_date`** / **`start`** *(str, default: `"2025-01-01"`)*: Startdatum (YYYY-MM-DD).
  * **`end_date`** / **`end`** *(str, optional, default: heute)*: Enddatum (YYYY-MM-DD).
  * **`budget`** *(float, default: `10000.0`)*: Positionsbudget in USD pro Trade.
  * **`clear_existing`** / **`clear`** *(bool, default: `true`)*: Vorherige Backfill-Trades dieser Strategie löschen.

### `POST /trades/backfill/<strategy_name>`
* **Beschreibung**: Historische Backfill-Simulation für eine spezifische Strategie über ein beliebiges Datumsintervall mit anpassbarem Budget. Unterstützte Strategien: `tgim`, `bridge_scout`, `bounce_bandit`.
* **Auth**: IP-Whitelist.
* **Pfad-Parameter**:
  * `strategy_name`: Strategie-Kennung (`tgim`, `bridge-scout`, `bounce-bandit`).
* **Query-Parameter (Datum & Budget)**:
  * **`start_date`** *(Alias `start`)*: `string` (Format: `YYYY-MM-DD`, Standard: `"2025-01-01"`). Beispiel: `start_date=2026-01-01`
  * **`end_date`** *(Alias `end`)*: `string` (Format: `YYYY-MM-DD`, Standard: aktueller Tag). Beispiel: `end_date=2026-06-30`
  * **`budget`**: `float` (Standard: `10000.0`). Handelsbudget in USD. Beispiel: `budget=5000.0`
  * **`clear_existing`** *(Alias `clear`)*: `bool` (Standard: `true`). Beispiel: `clear_existing=true`
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
      "closed_trades": []
    }
  }
  ```
* **Response `400 Bad Request`**: Bei Validierungsfehlern oder nicht für Backfill registrierten Strategien:
  ```json
  {
    "status": "error",
    "message": "Unknown strategy for backfill: 'dip_buyer'. Available strategies: ['tgim', 'thank_god_its_monday', 'bridge_scout', 'bridgescout', 'bridge scout', 'qqq_eom', 'bounce_bandit', 'bouncebandit', 'bounce bandit', 'qqq_meanrev']"
  }
  ```
* **Response `500 Internal Server Error`**: Bei serverseitigen Verarbeitungsfehlern.

---

## 7. Marktdaten-Synchronisation (`stocks.db`)

### `POST /market/sync`
* **Beschreibung**: Startet die inkrementelle oder vollständige Aktualisierung von Kursdaten (yfinance/TV) im Hintergrund.
* **Auth**: IP-Whitelist.
* **Query-Parameter / JSON Body**:
  * `full` *(bool, default: `false`)*: Bei `true` wird ein vollständiger Re-Download durchgeführt.
  * `provider` *(str, default: `"auto"`)*: Provider-Modus (`"auto"`, `"yahoo"`, `"tv"`).
  * `ignore_today` *(bool, default: `false`)*: Heutige Intraday-Bars ignorieren.
* **Response `202 Accepted`**:
  ```json
  {
    "status": "accepted",
    "message": "Sync started"
  }
  ```
* **Response `409 Conflict`**: Wenn bereits ein Synchronisationslauf im Hintergrund aktiv ist:
  ```json
  {
    "status": "error",
    "message": "Market synchronization already in progress"
  }
  ```

### `POST /market/reload`
* **Beschreibung**: Triggert einen vollständigen manuellen Re-Download aller Marktdaten im Hintergrund.
* **Auth**: IP-Whitelist.
* **Query-Parameter / JSON Body**:
  * `provider` *(str, default: `"auto"`)*: Provider-Modus (`"auto"`, `"yahoo"`, `"tv"`).
  * `ignore_today` *(bool, default: `false`)*: Heutige Intraday-Bars ignorieren.
* **Response `200 OK`**:
  ```json
  {
    "status": "queued",
    "message": "Full reload triggered"
  }
  ```
* **Response `409 Conflict`**: Wenn bereits ein Synchronisationslauf im Hintergrund aktiv ist:
  ```json
  {
    "status": "error",
    "message": "Market synchronization already in progress"
  }
  ```
