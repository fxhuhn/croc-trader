---
name: deadcode
description: Standardablauf zur systematischen Erkennung, Verifikation und Bereinigung von Dead Code und DRY-Verletzungen
trigger: /deadcode
domain: python-development
outputs:
  - dead_code_report
  - action_items
---

# Dead Code & DRY Workflow (/deadcode)

Dieses Dokument definiert den Standardablauf zur Identifikation, Verifikation und sicheren Beseitigung von verwaistem Code (Dead Code), ungenutzten Symbolen und schädlichen DRY-Verletzungen (Don't Repeat Yourself) in der Croc-Trader-Codebase. Ziel ist eine schlanke, performante und wartbare Codebasis unter strikter Einhaltung der System-Invarianten.

> [!IMPORTANT]
> **Voraussetzung vor jeder Ausführung:**
> Gemäß `.agents/AGENTS.md` müssen vor jeder Analyse oder Codeänderung zwingend:
> 1. **Step 1:** Beide Architekturdokumente gelesen werden (`architecture.md` & `references/architecture.md`).
> 2. **Step 2:** Die relevanten Skills aktiviert werden (`python-auditor`, bei Codeänderungen `python-craftsman` und `python-tester`).

---

## Phase 1: Automatisierte statische Analyse (Detection)

Zur Identifikation potenzieller Kandidaten werden die repo-eigenen Werkzeuge über das lokale Virtual Environment ausgeführt:

1. **Vulture-Scan (Dead Code & ungenutzte Symbole):**
   ```bash
   .venv/bin/vulture app/ scripts/ vulture_whitelist.py
   ```
   - *Konfiguration:* Beachtet die Schwellenwerte aus `pyproject.toml` (`min_confidence = 80`).
   - *Hinweis:* Ergebnisse von Vulture sind Kandidaten und keine finalen Beweise für Dead Code (Gefahr von False Positives bei dynamischen Aufrufen).

2. **Ruff Lint-Scan (Ungenutzte Imports, Variablen & auskommentierter Code):**
   ```bash
   .venv/bin/ruff check app/ --select F401,F841,B007,ERA001
   ```
   - `F401`: Unused imports (ungenutzte Modul-Importe)
   - `F841`: Unused local variables (ungenutzte lokale Variablen)
   - `B007`: Unused loop control variables (Schleifenvariablen ohne Nutzung)
   - `ERA001`: Commented-out code (auskommentierter Programmcode)

---

## Phase 2: Manuelle Verifikation & False-Positive-Ausschluss

Jeder Treffer aus Phase 1 muss zwingend gegen dynamische und deklarative Framework-Schnittstellen geprüft werden, bevor er als Dead Code deklariert wird:

1. **Webserver & Routen (`app/routes/`):**
   - Ist das Symbol ein Flask-View-Handler (`@blueprint.route`, `app.add_url_rule`)?
   - Wird die Funktion als Kontext-Prozessor oder Error-Handler genutzt?
2. **Frontend & Jinja2-Templates (`app/templates/`):**
   - Wird die Variable, Funktion oder Klasse in Jinja2-Templates referenziert?
   - Wird der Filter (`de_number`) oder das Makro dynamisch aus Templates aufgerufen?
3. **MCP-Server & Tools (`app/mcp/`):**
   - Ist das Symbol als MCP-Tool oder Resource registriert (`@mcp.tool()`)?
4. **Hintergrund-Jobs & Scheduler (`app/services/setup.py`):**
   - Wird die Methode als APScheduler-Job über Funktionsreferenz oder String-ID registriert?
5. **Trading-Strategien & Konstanten (`app/const.py`):**
   - Ist der Enum-Wert Teil der kanonischen `Strategies`-Enumeration oder des Strategie-Playbooks?
6. **Typ-Stubs & Dataclasses:**
   - Handelt es sich um Protocol-Properties, Pydantic-freie Dataclass-Attribute oder Schnittstellen-Signaturen?

---

## Phase 3: DRY-Analyse (Duplication vs. Orthogonality)

Prüfe redundante Codestellen anhand von `python.md §4.2`:

1. **Wissensduplikation vs. strukturelle Ähnlichkeit:**
   - Teilen zwei Codeblöcke dieselbe fachliche Geschäftslogik und ändern sie sich aus demselben Grund?
   - Wenn ja: Echte DRY-Verletzung.
   - Wenn nein: Zufällige Ähnlichkeit. Keine Abstraktion erzwingen.
2. **Kritische Redundanz-Zonen in Croc-Trader:**
   - **Indikatoren:** Mehrfache ATR-, SMA-, RSI- oder Momentum-Berechnungen außerhalb von `app/tools/indicators.py`.
   - **Datenbank & Repositories:** Redundante SQL-Query-Bausteine in `app/database/repositories/`.
   - **Position Sizing:** Duplizierte Cash- und Tick-Size-Rundungslogiken zwischen Screener und Trade Manager.
3. **Präferenz für lokale Duplikation:**
   - Eine falsche oder verfrühte Abstraktion ist schädlicher als kleine lokale Duplikation.
   - Eine Konsolidierung darf die Kopplung zwischen Modulen nicht unnötig erhöhen (Orthogonalität bewahren).

---

## Phase 4: Klassifizierung & Quality Pyramid

Ordne jedes verifizierte Finding in die Pyramide aus `python.md` ein:
1. **Layer 1: Correctness:** Führt die Beseitigung/Zusammenfassung zu Verhaltensänderungen oder Rundungsfehlern?
2. **Layer 2: Readability:** Verbessert die Bereinigung die Verständlichkeit (30-Sekunden-Regel)?
3. **Layer 3: Maintainability:** Sinkt die Wartungslast? Werden Schnittstellen stabiler?
4. **Layer 4: Changeability:** Werden zukünftige Modifikationen erleichtert?

---

## Phase 5: Output-Format (Audit-Bericht)

Der Bericht folgt dem Standard des `/review`-Workflows:

```markdown
# Code-Review Bericht: Dead Code & DRY Analysis

## Zusammenfassung
Executive Summary (Status: PASS/FAIL, Anzahl Dead-Code-Stellen, Anzahl DRY-Verletzungen, False Positives).

## Verifizierte Abweichungen (Findings)

### [DEAD-001 / DRY-001] Titel des Mangels
- **Datei/Symbol**: `Pfad:Zeile`
- **Klassifizierung**: `Pre-existing out of scope` | `Affected`
- **Schweregrad**: `High` | `Medium` | `Low`
- **Verletzte Regel/Dimension**: `z. B. Layer 3: Maintainability (python.md §4.2)`
- **Nachweis (Evidenz)**:
```python
# Relevanter Codeausschnitt
```
- **Verifikation**: Nachweis, warum kein False Positive vorliegt (z. B. "Symbol wird weder in Templates noch in Routen referenziert").
- **Korrekturvorgabe**: Konkrete Handlungsempfehlung (z. B. "Löschen", "In `vulture_whitelist.py` aufnehmen", "In `indicators.py` zusammenführen").

*(Falls keine Abweichungen vorliegen: "Keine ungenutzten Symbole oder kritischen DRY-Verletzungen festgestellt.")*

## Empfohlener Handlungsbedarf (Action Items)
- Priorisierte Liste der Bereinigungsschritte nach Risiko und Nutzen geordnet.
```

---

## Phase 6: Bereinigung & Regressionsschutz (falls beauftragt)

Falls der Anwender die Bereinigung autorisiert:
1. **Zero Unrequested Scope Expansion:** Nur die explizit genehmigten Symbole entfernen oder konsolidieren.
2. **Whitelist-Pflege:** Verifizierte dynamische Symbole (z. B. API-Responses, MCP-Tools) in `vulture_whitelist.py` eintragen statt Code künstlich zu verändern.
3. **Regressionsprüfung:**
   ```bash
   .venv/bin/pytest
   .venv/bin/ruff check .
   .venv/bin/mypy
   .venv/bin/pre-commit run --all-files
   ```
