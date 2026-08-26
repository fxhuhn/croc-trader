# Croc-Trader Test Suite

Dieses Verzeichnis enthält die automatisierte Test-Suite für die Croc-Trader-Plattform.

---

## 📁 Verzeichnisstruktur

Die Tests sind nach klaren Verantwortungsebenen strukturiert:

```
test/
├── conftest.py              # Globale Pytest-Fixtures & Mock-Wrapper
├── unit/                    # Isolidierte Modul- & Komponenten-Tests
│   ├── database/            # Repository-Tests (SQLite In-Memory)
│   ├── routes/              # Flask View & REST API Controller-Tests
│   ├── screener/            # Strategy Screener Engine Tests
│   ├── services/            # Service-Layer (Backfill, MarketUpdater, Quality)
│   ├── strategies/          # Strategie-Logik- & Entry/Exit-Tests
│   ├── tools/               # Indikatoren, Metriken & Symbol-Filter
│   └── trade_manager/       # TradeManager & Position Sizing Tests
├── integration/             # Komponententests mit integrierten Repositories
├── robustness/              # Stresstests, Boundary Checks & Error Resilience
└── security/                # Security Hardening, IP-Whitelisting & Error Routes
```

---

## 🚀 Test-Ausführung (Das 3-Stufen-Modell)

Die Testsuite ist in 3 Ausführungs-Tiers gegliedert:

### 1. Tier 1: Fast Gate (< 15s)
Führt alle isolierten Unit- und Boundary-Tests aus (Pre-Commit / schneller Entwickler-Loop):
```bash
.venv/bin/pytest -m tier1 -v
# oder alle Unit-Tests
.venv/bin/pytest test/unit/ -v
```

### 2. Tier 2: Verification Gate (< 2m)
Führt Property-Based Fuzzing (`hypothesis`), Lookahead-Bias-Guards und SQLite Chaos Injection aus:
```bash
.venv/bin/pytest -m tier2 -v
# oder gezielt die Robustness-Suite
.venv/bin/pytest test/robustness/ -v
```

### 3. Tier 3: Deep Hardening & Mutation Testing
Führt Mutation Testing auf dem Functional Core der Strategien aus:
```bash
.venv/bin/mutmut run
.venv/bin/mutmut results
```

### 4. Gesamte Suite mit Coverage
```bash
.venv/bin/pytest --cov=app --cov-report=term-missing
```

---

## 🛠️ Codequalität & Verification Gates

Vor jedem Commit / PR müssen die Verifikations-Gates erfüllt werden:

```bash
# 1. Code-Formatierung & Linting
.venv/bin/ruff format --check test/
.venv/bin/ruff check test/

# 2. Statische Typ-Prüfung (MyPy)
.venv/bin/mypy test/unit/
```

---

## 💡 Konventionen & Fixtures

1. **In-Memory SQLite**: Datenbank-Repository-Tests nutzen `DatabaseSession(str(tmp_path / "test.db"))` für schnelle, isolierte Ausführung ohne Festplatten-Seiteneffekte.
2. **Deterministic Time**: Zeitabhängige Tests nutzen feste Datumsstrings (z.B. `"2026-08-01"`).
3. **No External Network Access**: Externe APIs (yfinance, Telegram, IBKR) werden isoliert gemockt.
4. **Zero Lookahead-Bias**: Indikator- und Strategieberechnungen an Tag $T$ dürfen niemals zukünftige Daten einbeziehen.
5. **Dev-Only Dependencies**: `hypothesis` und `mutmut` sind strikt in `requirements-dev.txt` deklariert und existieren nicht im Docker-Produktiv-Image.

