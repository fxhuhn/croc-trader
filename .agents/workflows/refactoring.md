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
   - Ausführung von `.venv/bin/pytest` für das Zielmodul.
   - *Quality Gate 1:* Alle bestehenden Tests MÜSSEN grün sein. Bei fehlschlagenden Tests bricht der Workflow sofort ab.
2. **Coverage-Check & Test-Generierung:**
   - Abdeckung des Zielmoduls via `.venv/bin/pytest --cov=target_file` ermitteln.
   - *Quality Gate 2:* Ist die Coverage < 85 %, erstellt `python-tester` vorab temporäre *Characterization Tests* (in `tests/`, **niemals im Root-Verzeichnis**), um den aktuellen Zustand lückenlos abzusichern.

---

## Phase 2: Static Analysis & Security Audit (Diagnose)
**Verantwortliche Skills:** `python-auditor` & `python-security`

1. **Qualitäts-Scan (`python-auditor`):**
   - Syntax-, Style- und Type-Checks via `.venv/bin/ruff check .` und `.venv/bin/mypy`.
   - Prüfung gegen die Qualitäts-Pyramide (Correctness > Readability > Maintainability > Changeability).
   - Identifizierung von Code Smells (z. B. hohe zyklomatische Komplexität > 10, kognitive Komplexität > 15, mehr als 5 funktionale Parameter, fehlende `@dataclass(frozen=True)` / `TypedDict`, `print()` statt `logger`).
2. **Sicherheits-Scan (`python-security`):**
   - Auditierung des Moduls auf Schwachstellen (z. B. SQL-String-Interpolation, unsichere Handhabung von I/O, Credentials, float-basierte Währungsberechnungen).
3. **Erstellung des Audit-Protokolls:**
   - Konsolidierung aller Befunde in einer priorisierten Mängelliste (Teil A: Code Quality, Teil B: Security Findings).

---

## Phase 3: Iterative Transformation (Refactoring)
**Verantwortlicher Skill:** `python-craftsman` (`.agents/skills/python-craftsman`)

1. **Abarbeitung des Audit-Protokolls:**
   - Schrittweise Behebung der im Protokoll definierten Punkte.
   - Ersetzen von unflexiblen Dictionaries durch `@dataclass(frozen=True)` für immutable Domain-Objekte oder `TypedDict` für Datenstrukturen/DTOs (**Standard-Bibliothek first, kein Pydantic**).
   - Strikte Einhaltung des **synchronen EOD-Pipeline-Paradigmas** (kein `asyncio` / `async def`; saubere Entkopplung von I/O im Imperative Shell vom Pure Functional Core).
   - Einhalten strikter Separierung von Geschäftslogik und Datenzugriff (Functional Core / Imperative Shell).
   - Anwenden von Early-Return Guard Clauses zur Reduktion von Verschachtelungstiefen (max. 3 Ebenen).
2. **Inkrementelles Vorgehen:**
   - Anwendung von genau einer Refaktorierungskategorie pro Iterationsschritt (erst Typen/Validierung, dann Logik-Entkopplung).

---

## Phase 4: Final Verification & Quality Gates (Validierung)
**Verantwortliche Skills:** `python-tester`, `python-auditor`, `python-security`

1. **Regressionstest (`python-tester`):**
   - Erneutes Ausführen der gesamten Test-Suite via `.venv/bin/pytest`.
   - *Quality Gate 3:* Verhaltensäquivalenz gewahrt (100 % grün, 0 Regressionen).
2. **Gold Standard Compliance Gate (`python-auditor`):**
   - `.venv/bin/mypy` (0 Typ-Fehler im geprüften Scope).
   - `.venv/bin/ruff check .` (0 Linting-Warnungen).
   - `.venv/bin/ruff format --check .` (Formatierung konform).
3. **Security Gate (`python-security`):**
   - Verifikation von SQL-Parametrisierung, Decimal-Präzision und I/O-Sicherheit (0 kritische Befunde).

---

## Phase 5: Documentation & Diff Generation
**Verantwortlicher Skill:** `python-craftsman`

1. **Dokumentation:**
   - Aktualisierung oder Neuerstellung aller Docstrings im Google-Style für öffentliche Klassen und Funktionen.
2. **Zusammenfassung (Completion Report):**
   - Erstellung des standardisierten Abschlussberichts gemäß `.agents/AGENTS.md`:
     - `Changed`: Umgesetztes Ergebnis.
     - `Files`: Modifizierte Dateien.
     - `Validation`: Ausgeführte Befehle & Ergebnisse.
     - `Not validated`: Nicht ausführbare Prüfungen.
     - `Assumptions`: Verbleibende Annahmen.
     - `Out of scope`: Relevante Befunde außerhalb des Tasks.
