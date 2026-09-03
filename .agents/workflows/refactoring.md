---
name: refactor
description: Standardablauf für die sichere, schrittweise und automatisierte Umstrukturierung von Python-Code (Refactoring Engine)
trigger: manual
domain: python-development
inputs:
  - target_file_path
  - refactoring_goal # z. B. "type-safety", "architecture", "performance", "security"
outputs:
  - refactored_code
  - audit_report
  - verification_summary
---

# Refactoring Workflow (Automated Python Refactoring Engine)

Dieses Dokument definiert den Standardablauf für die sichere, schrittweise und automatisierte Umstrukturierung von bestehendem Python-Code zur Erreichung des Repository-Gold-Standards (strikte Typisierung, saubere Architektur, hohe Testabdeckung), ohne das bestehende Verhalten zu verändern (Behavioral Equivalence).

> [!IMPORTANT]
> **Voraussetzung für die Ausführung:**
> Vor jeder Code-Analyse oder -Modifikation MÜSSEN die obligatorischen Schritte aus `.agents/AGENTS.md` eingehalten werden (Step 1: Architektur-Inspektion von `architecture.md` & `references/architecture.md`, Step 2: Skill-Aktivierung).

---

## Phase 1: Baseline Verification (Status Quo sichern)
**Verantwortlicher Skill:** `python-tester` (`.agents/skills/python-tester`)

1. **Bestehende Test-Suite ausführen:**
   - Ausführung von `.venv/bin/pytest` für das Zielmodul und direkt gekoppelte Subsysteme.
   - *Quality Gate 1:* Alle bestehenden Tests MÜSSEN grün sein. Bei fehlschlagenden Tests bricht der Workflow sofort ab.
2. **Branch-Coverage & Varianten-Matrix:**
   - Abdeckung via `.venv/bin/coverage run -m pytest` und `.venv/bin/coverage report -m target_file` ermitteln (inklusive Branch-Coverage!).
   - Nicht nur auf pauschale Zeilen-Coverage blicken: Systematische Prüfung gegen eine **fachliche Varianten-Matrix** (Grenzfälle, Verzweigungen, Null-Werte, optionale Parameter).
3. **Dauerhafte Characterization Tests (Pinning Tests):**
   - Für alle ungedeckten Pfade und Zweige MÜSSEN vorab Characterization-Tests in `test/` erstellt werden.
   - *Wichtig:* Diese Tests sind **dauerhafter Regressionsschutz** und dürfen nach dem Refactoring **nicht** gelöscht werden.
4. **Golden-Master- / Snapshot-Sicherung (bei I/O & Datenexporten):**
   - Erzeugt das Modul persistente Artefakte (z. B. CSV, JSON, Berichte), wird vor dem Code-Eingriff ein Referenz-Snapshot (Golden Master) der Ausgabe erzeugt, um nach dem Refactoring absolute Byte- und Inhalts-Gleichheit nachzuweisen.
   - *Quality Gate 2:* Branch-Coverage $\ge 90\,\%$, alle Varianten der Matrix erfasst, Snapshot gesichert.

---

## Phase 2: Static Analysis, Security & Blast-Radius Audit (Diagnose)
**Verantwortliche Skills:** `python-auditor` & `python-security`

1. **Blast-Radius & Aufrufer-Analyse:**
   - Ermittlung aller externen Aufrufer und Konsumenten der öffentlichen Funktionen/Klassen via `grep_search`.
   - Sicherstellen, dass geplante Signaturänderungen rückwärtskompatibel sind oder alle Aufrufer migriert werden.
2. **Qualitäts-Scan (`python-auditor`):**
   - Syntax-, Style- und Type-Checks via `.venv/bin/ruff check .` und `.venv/bin/mypy`.
   - Prüfung gegen die Qualitäts-Pyramide (Correctness > Readability > Maintainability > Changeability).
   - Identifizierung von Code Smells (z. B. zyklomatische Komplexität > 10, kognitive Komplexität > 15, mehr als 5 funktionale Parameter, fehlende `@dataclass(frozen=True)` / `TypedDict`, `print()` statt `logger`).
3. **Sicherheits- & I/O-Scan (`python-security`):**
   - Auditierung auf Schwachstellen (z. B. SQL-String-Interpolation, unsichere Pfad-Handhabung / Magic Paths, unvollständige I/O-Isolation in Tests, float-basierte Währungsberechnungen).
4. **Erstellung des Audit-Protokolls:**
   - Konsolidierung aller Befunde in einer priorisierten Mängelliste (Teil A: Code Quality, Teil B: Security & Boundaries).

---

## Phase 3: Iterative Transformation (Refactoring)
**Verantwortlicher Skill:** `python-craftsman` (`.agents/skills/python-craftsman`)

1. **Abarbeitung des Audit-Protokolls in Micro-Schritten:**
   - Schrittweise Behebung der im Protokoll definierten Punkte.
   - **Micro-Test-Schleife (TDD-Refactoring):** Nach *jedem einzelnen* Teilschritt (z. B. Extraktion einer Konstante, Umstellung eines Lookups) wird die Test-Suite sofort ausgeführt (`Edit` $\rightarrow$ `Tests grün` $\rightarrow$ `nächster Edit`).
   - Schlägt ein Zwischenschritt fehl, wird sofort auf den letzten grünen Stand zurückgerollt.
2. **Architektur- und Clean-Code-Regeln:**
   - Ersetzen von unflexiblen Dictionaries durch `@dataclass(frozen=True)` für immutable Domain-Objekte oder `TypedDict` für Datenstrukturen/DTOs (**Standard-Bibliothek first, kein Pydantic**).
   - Strikte Einhaltung des **synchronen EOD-Pipeline-Paradigmas** (kein `asyncio` / `async def`; saubere Entkopplung von I/O im Imperative Shell vom Pure Functional Core).
   - Dependency Injection statt harter Kopplung an globale Singletons (`settings`).
   - Anwenden von Early-Return Guard Clauses zur Reduktion von Verschachtelungstiefen (max. 3 Ebenen).
3. **Inkrementelle Reihenfolge:**
   - Schritt A: Konstanten, Typen & DTOs extrahieren.
   - Schritt B: Interne Logik, Algorithmen & Formatierer entkoppeln (Pure Functions).
   - Schritt C: I/O, Pfade & Schnittstellen harmonisieren.

---

## Phase 4: Final Verification & Quality Gates (Validierung)
**Verantwortliche Skills:** `python-tester`, `python-auditor`, `python-security`

1. **Regressionstest & Snapshot-Vergleich (`python-tester`):**
   - Erneutes Ausführen der gesamten Test-Suite via `.venv/bin/pytest`.
   - Bei I/O-Modulen: Byte-für-Byte- oder semantischer Diff-Vergleich mit dem Golden-Master-Snapshot aus Phase 1.
   - *Quality Gate 3:* Verhaltensäquivalenz zu 100 % gewahrt (0 Regressionen, Snapshot identisch).
2. **Gold Standard Compliance Gate (`python-auditor`):**
   - `.venv/bin/mypy` (0 Typ-Fehler im geprüften Scope).
   - `.venv/bin/ruff check .` (0 Linting-Warnungen).
   - `.venv/bin/ruff format --check .` (Formatierung konform).
3. **Security Gate (`python-security`):**
   - Verifikation von SQL-Parametrisierung, Decimal-Präzision und I/O-Sicherheit (0 kritische Befunde).

---

## Phase 5: Documentation & Diff Generation
**Verantwortlicher Skill:** `python-craftsman`

1. **Diff-Inspektion (`git diff`):**
   - Strikte Prüfung gegen Scope Leaks oder unabsichtliche Formatierungsänderungen außerhalb des Refactorings.
2. **Dokumentation:**
   - Aktualisierung oder Neuerstellung aller Docstrings im Google-Style für öffentliche Klassen und Funktionen.
3. **Zusammenfassung (Completion Report):**
   - Erstellung des standardisierten Abschlussberichts gemäß `.agents/AGENTS.md`:
     - `Changed`: Umgesetztes Ergebnis.
     - `Files`: Modifizierte Dateien.
     - `Validation`: Ausgeführte Befehle & Ergebnisse.
     - `Not validated`: Nicht ausführbare Prüfungen.
     - `Assumptions`: Verbleibende Annahmen.
     - `Out of scope`: Relevante Befunde außerhalb des Tasks.
