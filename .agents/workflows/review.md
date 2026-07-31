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
