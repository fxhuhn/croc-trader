# Croc-Trader Test Suite

Dieses Verzeichnis enthält die automatisierte Test-Suite für die Croc-Trader-Plattform.

---

## 📁 Verzeichnisstruktur

Die Tests sind nach klaren Verantwortungsebenen strukturiert:

```
test/
├── conftest.py              # Globale Pytest-Fixtures & Mock-Wrapper
├── unit/                    # Isolidierte Modul- & Komponenten-Tests
│   ├── backtester/          # Unit-Tests für Backtester-Math & Analytics
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

## 🚀 Test-Ausführung

### 1. Gesamten Test-Suite ausführen
```bash
.venv/bin/pytest
```

### 2. Test-Suite mit Coverage-Report ausführen
```bash
.venv/bin/pytest --cov=app --cov-report=term-missing
```

### 3. Spezifischen Test-Ebenen ausführen
```bash
# Nur Unit-Tests ausführen
.venv/bin/pytest test/unit/ -v

# Nur Security- & Hardening-Tests ausführen
.venv/bin/pytest test/security/ -v

# Einzelne Testdatei ausführen
.venv/bin/pytest test/unit/database/repositories/test_trade_repository.py -v
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
