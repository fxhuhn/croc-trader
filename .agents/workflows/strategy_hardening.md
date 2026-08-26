---
description: Standardablauf zur mathematischen und fehlerresistenten Härtung von Handelsstrategien (BVA, Hypothesis Fuzzing, Lookahead Guards, Mutation Testing, Concurrency Resilience)
---

# Workflow: Strategy Hardening & Anti-Fragility

Dieser Workflow dient der systematischen Härtung einer Screener- oder Trade-Manager-Strategie gegen unvorhergesehene Marktanomalien, Datenlecks und Ausführungsfehler.

---

## Ablauf in 5 Schritten

### Schritt 1: Boundary Value Analysis (BVA) & Negative Tests (Tier 1)
1. Erstelle oder erweitere `test/robustness/test_strategy_boundaries.py` bzw. `test/unit/strategies/test_<strategy>_boundaries.py`.
2. Prüfe zwingend folgende Grenzfälle ab:
   * **OHLCV-Ränder:** Leere Historie, $N < \text{Lookback}$, $High == Low == Open == Close$, $Volume == 0$, Flash-Crashes / Gaps $> 50\,\%$.
   * **Null-Volatilität:** $ATR = 0$ darf unter keinen Umständen eine `ZeroDivisionError` oder unhandled Exception auslösen.
   * **Kollisions-Priorität:** Bar berührt gleichzeitig Stop-Loss und Take-Profit $\rightarrow$ Verifikation des Worst-Case-Exits.
   * **Order-Sizing:** $Size = 0$, $Kapital = 0$, Pennystock-Preise ($0.001$).
3. **Gate:** `.venv/bin/pytest -m tier1` muss in $< 15$ Sekunden erfolgreich durchlaufen.

---

### Schritt 2: Property-Based Fuzzing mit Hypothesis (Tier 2)
1. Definiere mathematische Invarianten für den Functional Core der Strategie in `test/robustness/test_strategy_invariants.py`.
2. Generiere randomisierte OHLCV-Serien mit `@given(...)`.
3. Verifiziere:
   * Keine ungefangenen Exceptions (`KeyError`, `IndexError`, `ZeroDivisionError`).
   * Alle Indikatorwerte liegen im definierten Intervall (z. B. $0 \le RSI \le 100$).
   * Stops liegen bei Long-Setups immer strikt unter dem Entry-Preis.
4. **Gate:** `.venv/bin/pytest test/robustness/test_strategy_invariants.py` muss $50+$ randomisierte Testfälle fehlerfrei bestehen.

---

### Schritt 3: Zero Lookahead-Bias & Engine-Parity Guard (Tier 2)
1. Verifiziere in `test/robustness/test_lookahead_bias.py`:
   * Point-in-Time-Invariante: Signal/Indikator auf Bar $T$ ist identisch, egal ob Daten bis $T$ oder $T+N$ vorliegen.
   * Keine zukunftsorientierten Shift-/Rolling-Operationen.
2. **Gate:** `.venv/bin/pytest test/robustness/test_lookahead_bias.py` muss bestehen.

---

### Schritt 4: Chaos & Fault Injection Verification (Tier 2)
1. Teste die Strategie-Ausführung unter SQLite-Locks und Transaktionsfehlern (`test/robustness/test_sqlite_chaos.py`).
2. Prüfe die Idempotenz: Wiederholter Lauf für dasselbe Datum erzeugt 0 Duplikate.
3. **Gate:** `.venv/bin/pytest test/robustness/test_sqlite_chaos.py` muss bestehen.

---

### Schritt 5: Mutation Testing & Typ-Härtung (Tier 3)
1. Führe `mutmut` auf das Strategiemodul aus:
   ```bash
   .venv/bin/mutmut run --paths-to-mutate=app/services/screener/strategies/<strategy>.py,app/services/trade_manager/strategies/<strategy>.py
   .venv/bin/mutmut results
   ```
2. Analysiere überlebende Mutanten und ergänze fehlende Assertions (Ziel: Mutation Score $\ge 85\,\%$).
3. Entferne das Strategiemodul aus `[[tool.mypy.overrides]]` in `pyproject.toml`, um strikte Typ-Prüfung zu aktivieren:
   ```bash
   .venv/bin/mypy app/services/screener/strategies/<strategy>.py
   ```

---

## Abschlusskriterien (Quality Gate)
- [ ] Alle Tier-1- und Tier-2-Tests grün.
- [ ] Keine neuen MyPy-Fehler im gehärteten Modul.
- [ ] Ruff Format & Lint fehlerfrei: `.venv/bin/ruff check .` und `.venv/bin/ruff format --check .`.
