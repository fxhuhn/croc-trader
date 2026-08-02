---
name: review-workflow
description: Standardablauf für Code-Audits, Qualitätsprüfungen und Security-Reviews
trigger: manual
---

# Review Workflow

Dieses Dokument definiert den Standardablauf für Code-Audits und Reviews.

## 1. Domain-Review (Optional)
Falls zutreffend, wird der domänenspezifische Skill geladen:
- Strategien: `strategy-screener`
- Python-Code: `python-auditor`
- Security/Data-Integrity: `python-security`
- Frontend: `flask-ui`
- Datenqualität: `data-ingestion`

## 2. Qualitätsprüfung
Der Code wird gegen die Pyramide aus `python.md` geprüft:
- **Correctness:** Werden Randfälle und Fehler abgefangen?
- **Readability:** Stimmen Benennungen und Längen?
- **Maintainability:** Ist die Typisierung vollständig (Mypy)?
- **Changeability:** Sind Kapselung und SOLID (wo sinnvoll) eingehalten?

## 3. Output-Format (Review-Bericht)

Der Review-Bericht muss zwingend folgendes Format einhalten:

```markdown
# Code-Review Bericht

## Zusammenfassung
Short Executive Summary (Status: PASS/FAIL, Anzahl blockierender Findings).

## Gefundene Abweichungen (Findings)

### [Finding-ID] Titel des Mangels
- **Datei/Symbol**: `Pfad:Zeile`
- **Klassifizierung**: `Introduced` | `Affected` | `Pre-existing out of scope`
- **Schweregrad**: `Blocking` | `High` | `Medium` | `Low`
- **Verletzte Regel/Dimension**: `z. B. Layer 1: Correctness`
- **Nachweis (Evidenz)**: Exakter Ausschnitt des problematischen Codes.
- **Korrekturvorgabe**: Konkrete Anweisung zur Behebung.

*(Falls keine Abweichungen vorliegen: "Keine blockierenden Mängel festgestellt.")*

## Empfohlener Handlungsbedarf (Action Items)
- Punkt 1...
```
