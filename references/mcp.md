# Croc-Trader Model Context Protocol (MCP) Reference

Dokumentation der MCP-Schnittstelle von Croc-Trader (`app/routes/mcp.py`, `app/mcp/`).

---

## 🔒 Authentifizierung & Endpunkt

* **Protokoll**: Streamable HTTP / JSON-RPC 2.0
* **Endpunkt-URL**: `POST http://<host>:<port>/mcp` (Standard: `http://127.0.0.1:8001/mcp`)
* **IP-Whitelist Schutz**: Gesichert über den Decorator `@require_ip_whitelist` (Konfiguration in `security.whitelist`).
* **Content-Type**: `application/json` (UTF-8).
* **Deaktivierter Server**: Ist `mcp.enabled: false` gesetzt oder `mcp_server` nicht initialisiert, liefert der Endpunkt `503 Service Unavailable`.

---

## ⚙️ Einrichtung & Konfiguration

### 1. Server-Aktivierung in `settings.yaml`

In `settings.yaml` muss der Abschnitt `mcp` aktiviert sein:

```yaml
mcp:
  enabled: true

webserver:
  host: 127.0.0.1
  port: 8001
  debug: false

security:
  mode: warning
  whitelist:
    - "127.0.0.1"
```

### 2. Server starten

```bash
.venv/bin/python run.py
```

### 3. Client-Konfiguration

#### Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "croc-trader": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

#### Antigravity / Cursor IDE (`.agents/plugins/croc-mcp/mcp_config.json`)

```json
{
  "mcpServers": {
    "croc-trader-http": {
      "url": "http://127.0.0.1:8001/mcp"
    }
  }
}
```

---

## 📡 JSON-RPC 2.0 Basismethoden

Der Endpunkt akzeptiert standardkonforme JSON-RPC 2.0 Anfragen:

| Methode | Beschreibung | Parameter |
| :--- | :--- | :--- |
| **`initialize`** | Handshake & Versionsabgleich | `protocolVersion`, `capabilities`, `clientInfo` |
| **`ping`** | Verbindungsprüfung / Heartbeat | Keine |
| **`tools/list`** | Auflistung aller registrierten MCP-Tools | Keine |
| **`tools/call`** | Ausführung eines spezifischen Tools | `name` *(str)*, `arguments` *(dict)* |
| **`resources/list`** | Auflistung aller URI-Ressourcen | Keine |
| **`resources/read`** | Auslesen einer spezifischen Ressource | `uri` *(str)* |
| **`prompts/list`** | Auflistung aller vordefinierten Prompt-Templates | Keine |
| **`prompts/get`** | Abrufen eines befüllten Prompt-Templates | `name` *(str)*, `arguments` *(dict)* |
| **`notifications/*`** | Einweg-Benachrichtigungen (z. B. `notifications/initialized`) | Liefert HTTP `204 No Content` |

---

## 🛠️ Tool-Referenz (20 Domain- & Aktions-Tools)

### 1. Portfolio & Positionsüberwachung (`app/mcp/tools/portfolio.py`)

#### `get_active_positions`
* **Beschreibung**: Liefert alle aktuell offenen Positionen aus `signals.db` mit aktuellem Kurs, Stop-Loss, Take-Profit und unrealisiertem PnL.
* **Argumente**:
  * `strategy` *(str, optional)*: Filter nach Strategiename (z. B. `"dip_buyer"`, `"tgim"`, `"ndx_momentum"`).

#### `get_portfolio_summary`
* **Beschreibung**: Aggregierte Portfolio-Statistiken: Gesamtes investiertes Kapital, offener PnL und Aufschlüsselung nach Strategien.
* **Argumente**: Keine.

#### `get_trade_history`
* **Beschreibung**: Historie geschlossener Trades mit Haltedauer, Ausstiegsgrund und realisiertem PnL.
* **Argumente**:
  * `strategy` *(str, optional)*: Filter nach Strategiename.
  * `limit` *(int, optional, Standard: `50`, Max: `200`)*: Maximale Anzahl an Datensätzen.

#### `get_trade_detail`
* **Beschreibung**: Vollständiger Datensatz eines einzelnen Trades inklusive vollständigem Lifecycle-Audit-Log (`trade_logs`).
* **Argumente**:
  * `trade_id` *(int, erforderlich)*: ID des Trades in `signals.db`.

---

### 2. Screener & Signale (`app/mcp/tools/screener.py`)

#### `get_screener_candidates`
* **Beschreibung**: Setup-Kandidaten aus den Screening-Strategien (DipBuyer, NDXMomentum, TGIM, BridgeScout, BounceBandit, HoldTarget, SplitTarget).
* **Argumente**:
  * `strategy` *(str, optional)*: Kanonischer Strategiename.
  * `limit` *(int, optional, Standard: `50`, Max: `200`)*: Maximale Anzahl an Kandidaten.

#### `get_turnover_candidates`
* **Beschreibung**: Aggregierte Turnover-Timing-Kandidaten, sortiert nach dem 20-Tage Average Dollar Volume.
* **Argumente**:
  * `limit` *(int, optional, Standard: `50`, Max: `200`)*: Maximale Anzahl an Datensätzen.

#### `get_webhook_signals`
* **Beschreibung**: Zuletzt empfangene TradingView-Webhook-Signale aus der `croc`-Tabelle.
* **Argumente**:
  * `symbol` *(str, optional)*: Ticker-Symbol Filter (z. B. `"TSLA"`).
  * `limit` *(int, optional, Standard: `20`, Max: `100`)*: Maximale Anzahl an Signalen.

---

### 3. Marktdaten & Symbol-Universum (`app/mcp/tools/market.py`)

#### `get_price_history`
* **Beschreibung**: Historische tägliche OHLCV-Candlesticks für ein Symbol aus `stocks.db`.
* **Argumente**:
  * `symbol` *(str, erforderlich)*: Aktien-Ticker (z. B. `"AAPL"`, `"QQQ"`).
  * `days` *(int, optional, Standard: `60`, Max: `500`)*: Anzahl zurückliegender Bars.

#### `get_latest_prices`
* **Beschreibung**: Zuletzt gespeicherter Schlusskurs für eine Liste von Symbolen.
* **Argumente**:
  * `symbols` *(list[str], erforderlich)*: Liste von Ticker-Symbolen (z. B. `["AAPL", "MSFT"]`).

#### `get_symbol_universe`
* **Beschreibung**: Übersicht aller getrackten Symbole in `stocks.db` sowie der aktuellen Blacklist (`ignored_symbols`).
* **Argumente**: Keine.

---

### 4. Broker & Orders (`app/mcp/tools/orders.py`)

#### `get_broker_active_orders`
* **Beschreibung**: Aktive Broker-Orders (ENTRY, STOP LOSS, TAKE PROFIT) aus `trading.db`.
* **Argumente**:
  * `status` *(str, optional)*: Filter nach Status (z. B. `"Submitted"`, `"PreSubmitted"`, `"Created"`).

#### `get_broker_settlements`
* **Beschreibung**: Abgewickelte Trade-Gruppen mit Ausführungs-VWAP, Slippage gegenüber Signalpreis, Kommissionen und Netto-PnL.
* **Argumente**:
  * `limit` *(int, optional, Standard: `50`, Max: `200`)*: Maximale Anzahl.

#### `list_order_csv_files`
* **Beschreibung**: Liste generierter CSV-Orderdateien im Verzeichnis `data/orders/` mit Dateigröße und Änderungsdatum.
* **Argumente**: Keine.

---

### 5. System & Konfiguration (`app/mcp/tools/system.py`)

#### `get_system_health`
* **Beschreibung**: Diagnosestatus aller SQLite-Datenbanken (`stocks.db`, `signals.db`, `trading.db`), Dateigrößen, Zeitstempel der letzten Kurs-/Trade-Aktualisierung und Scheduler-Status.
* **Argumente**: Keine.

#### `get_strategy_list`
* **Beschreibung**: Liste aller im System registrierten Handelsstrategien (`Strategies` Enum) mit ihren konfigurierten Allokationsbudgets und Risikobeträgen aus `settings.yaml`.
* **Argumente**: Keine.

---

### 6. Aktionen & Workflow-Steuerung (`app/mcp/tools/actions.py`)

#### `trigger_screener`
* **Beschreibung**: Führt den Screener für alle aktiven Strategien oder eine spezifische Strategie synchron aus und speichert Treffer in `signals.db`.
* **Argumente**:
  * `strategy_name` *(str, optional)*: Strategiename (z. B. `"dip_buyer"`, `"two_percent"`).
  * `lookback_days` *(int, optional, Standard: `0`)*: Lookback-Offset in Handelstagen (`0` = neuester Datensatz).

#### `trigger_single_symbol_debug`
* **Beschreibung**: Führt eine detaillierte Regelprüfung für ein einzelnes Symbol gegen eine Strategie aus.
* **Argumente**:
  * `strategy_name` *(str, erforderlich)*: Strategiename (z. B. `"dip_buyer"`, `"turnover_timing"`).
  * `symbol` *(str, erforderlich)*: Ticker-Symbol (z. B. `"AAPL"`).

#### `trigger_order_generation`
* **Beschreibung**: Berechnet die täglichen Bracket-Orders für alle aktiven Positionen und neue Setups und exportiert die CSV-Orderdatei nach `data/orders/`.
* **Argumente**: Keine.

#### `trigger_eod_pipeline`
* **Beschreibung**: Führt die vollständige tägliche EOD-Pipeline sequentiell aus (TradeManager $\rightarrow$ Screener $\rightarrow$ Orders $\rightarrow$ Cache Pre-warming).
* **Argumente**: Keine.

#### `trigger_strategy_backfill`
* **Beschreibung**: Führt eine historische Simulation für eine Strategie durch und speichert die Trades in `signals.db`.
* **Argumente**:
  * `strategy_name` *(str, erforderlich)*: Strategie-Bezeichner (z. B. `"tgim"`, `"bridge_scout"`).
  * `start_date` *(str, optional, Standard: `"2025-01-01"`)*: Simulationsstart (`YYYY-MM-DD`).
  * `end_date` *(str, optional)*: Simulationsende (`YYYY-MM-DD`).
  * `budget` *(float, optional, Standard: `10000.0`)*: Handelskapital in USD.
  * `clear_existing` *(bool, optional, Standard: `true`)*: Vorherige Backtest-Trades dieser Strategie löschen.

---

## 📦 Resource-Referenz

Ressourcen bieten passive JSON-Snapshots, die über standardisierte URIs abgerufen werden können:

| URI | Name | Beschreibung |
| :--- | :--- | :--- |
| **`croc://portfolio/active`** | `active_positions` | JSON-Snapshot aller aktuell offenen Positionen |
| **`croc://portfolio/summary`** | `portfolio_summary` | Gesamtbilanz: Strategien, Budgets, gebundenes Kapital, offener PnL |
| **`croc://system/health`** | `system_health` | Datenbankstatus, Dateigrößen und Aktualitätszeitstempel |
| **`croc://strategies`** | `strategy_registry` | Vollständige Strategieliste inklusive Allokationsbudgets |

---

## 💬 Prompt-Template Referenz

Vordefinierte Analyse-Prompts für KI-Workflows:

| Prompt-Name | Argumente | Zweck |
| :--- | :--- | :--- |
| **`daily-briefing`** | Keine | Umfassende EOD-Analyse: Portfolio-Status, offenes Risiko, neue Screener-Setups und Daten-Health. |
| **`trade-post-mortem`** | `trade_id` *(int)* | Forensische Analyse eines geschlossenen Trades anhand der Parameter, Lifecycle-Logs und Exit-Gründe. |
| **`strategy-review`** | `strategy_name` *(str)* | Statistische Performance-Auswertung einer Strategie (Win-Rate, Haltedauer, PnL-Verteilung). |

---

## 💻 Request- & Response-Beispiele (cURL)

### 1. Handshake (`initialize`)

```bash
curl -s -X POST http://127.0.0.1:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {"name": "curl-client", "version": "1.0"}
    }
  }'
```

### 2. Tool-Aufruf (`tools/call`: Aktive Positionen abrufen)

```bash
curl -s -X POST http://127.0.0.1:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "get_active_positions",
      "arguments": {
        "strategy": "dip_buyer"
      }
    }
  }'
```

### 3. Resource abrufen (`resources/read`)

```bash
curl -s -X POST http://127.0.0.1:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/read",
    "params": {
      "uri": "croc://portfolio/summary"
    }
  }'
```

### 4. Screener ausführen (`tools/call`: `trigger_screener`)

```bash
curl -s -X POST http://127.0.0.1:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "trigger_screener",
      "arguments": {
        "strategy_name": "dip_buyer",
        "lookback_days": 0
      }
    }
  }'
```

---

## ⚠️ Fehlercodes (JSON-RPC 2.0 & HTTP)

| Code | Typ | Ursache |
| :--- | :--- | :--- |
| **`-32700`** | Parse error | Request-Body ist kein gültiges JSON. |
| **`-32600`** | Invalid Request | Pflichtfelder (z. B. `method`) fehlen. |
| **`-32601`** | Method not found | Unbekannte JSON-RPC Methode aufgerufen. |
| **`-32602`** | Invalid params | Fehlende Pflichtparameter (z. B. `name` bei `tools/call`). |
| **`-32603`** | Internal error | Unerwarteter interner Serverfehler bei der Verarbeitung. |
| **`403`** | HTTP Forbidden | Client-IP steht nicht auf der `security.whitelist`. |
| **`503`** | Service Unavailable | MCP Domain Server ist in `settings.yaml` deaktiviert (`mcp.enabled: false`). |
