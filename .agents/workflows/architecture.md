---
name: architecture-workflow
description: Standardablauf für Architekturänderungen (Spezifikation, Handoff & Implementierung)
trigger: manual
---

# Architecture Workflow

Dieses Dokument definiert den Standardablauf für Architekturänderungen.

## 1. Initiale Spezifikation (architecture-specification)
Der Agent nutzt den Skill `architecture-specification`, um:
- Anforderungen zu validieren,
- den Scope gegen die Bestandsarchitektur zu prüfen,
- notwendige Diagramme (Mermaid) zu erstellen, falls hilfreich,
- einen Blueprint und Akzeptanzkriterien zu verfassen.

## 2. Übergabe an die Implementierung (python-craftsman)
Der Agent wechselt zum Skill `python-craftsman`, um:
- den Blueprint iterativ umzusetzen,
- Repository-Regeln (python.md, workspace.md) durchzusetzen.

## 3. Verifikation & Review
Der Agent führt die Verification Gates aus `python-craftsman` durch:
- Gate 3: Pytest (`python-tester`)
- Gate 4: Quality Audit (`python-auditor`), bei Bedarf
- Gate 5: Security Audit (`python-security`), bei Bedarf
- Gate 6: Architecture Sync (`architecture-sync`), bei Bedarf
