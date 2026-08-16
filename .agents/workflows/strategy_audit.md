---
name: strategy-audit
description: Standardablauf für das Auditieren und Abgleichen von Handelsstrategien gegen ihre kanonischen Playbooks
trigger: manual
domain: strategy-screening
inputs:
  - strategy_name # z. B. "dip_buyer", "two_percent", "bounce_bandit"
outputs:
  - audit_report
  - findings_matrix
---

# Strategy Audit Workflow

Dieses Dokument definiert den Standardablauf für die vollständige, schreibgeschützte Auditierung einer Handelsstrategie. Der Workflow gleicht das normative Strategy Playbook (`.agents/skills/strategy-screener/playbook/`) gegen die tatsächliche Implementierung im Screener, Trade Manager, den Konfigurationen und Tests ab.

> [!IMPORTANT]
> **Voraussetzung für die Ausführung:**
> Vor jeder Analyse MÜSSEN die obligatorischen Schritte aus `.agents/AGENTS.md` eingehalten werden (Step 1: Architektur-Inspektion von `architecture.md` & `references/architecture.md`, Step 2: Skill-Aktivierung `strategy-screener`).

---

## Phase 1: Evidence Gathering (Quelleninspektion)
**Verantwortlicher Skill:** `strategy-screener` (`.agents/skills/strategy-screener`)

1. **Playbook & Globale Verträge laden:**
   - Laden des kanonischen Playbooks `.agents/skills/strategy-screener/playbook/<strategy_name>.md`.
   - Laden von `.agents/skills/strategy-screener/playbook/overview.md` für geteilte State- und Lifecycle-Definitionen.
2. **Quellcode- & Test-Inspektion:**
   - Screener-Strategie: `app/services/screener/strategies/<strategy_name>.py`
   - Trade-Manager-Logik: `app/services/trade_manager/strategies/<strategy_name>.py` (bzw. `manager.py`)
   - Konfiguration: `settings.yaml` (unter `strategies.<strategy_name>`)
   - Unittests: `test/unit/screener/` und `test/unit/trade_manager/`
3. **Status:** Keine Code-Modifikationen. Alle Erkenntnisse basieren auf verifizierten Repository-Belegen.

---

## Phase 2: 4-Ebenen-Matrix-Abgleich (Soll vs. Ist)
**Verantwortlicher Skill:** `strategy-screener`

Die Strategie wird entlang von 4 Prüfelementen detailliert abgeglichen:

1. **Ebene 1: Indikatoren & Parameter**
   - Stimmen Lookback-Perioden (z. B. RSI-Länge, ATR-Perioden, SMA-Filter) zwischen Playbook und Python-Code überein?
   - Sind Schwellenwerte hardcodiert oder konfigurierbar über `settings.yaml`?
2. **Ebene 2: Screener Stage & Entry-Bedingungen**
   - Werden alle Universums-Filter, Mindestvolumen und Setup-Trigger exakt wie im Playbook abgebildet?
   - Stimmt die Ranking- und Priorisierungslogik bei mehreren Signalen?
3. **Ebene 3: Trade Manager Stage (Sizing & Exit-Regeln)**
   - Stimmen Risikomodell, Positionsgrößenberechnung, Initial-Stop-Loss, Target-Preise und Trailing-Stops überein?
   - Werden Zeit-basierte Exits (z. B. Max-Holding-Days) korrekt evaluiert?
4. **Ebene 4: Lifecycle, Status & Order-Mapping**
   - Werden die Statusübergänge (`CREATED` $\rightarrow$ `ACTIVE` $\rightarrow$ `INVALIDATED` / `CLOSED`) strikt eingehalten?
   - Stimmt das Order-Mapping (z. B. `order_type='MKT'` mit `tif='OPG'` für Market-On-Open) mit `references/architecture.md` überein?

---

## Phase 3: Klassifizierung & Befundungsmatrix
**Verantwortlicher Skill:** `strategy-screener`

Jeder Teilaspekt wird anhand des kanonischen Vokabulars eingestuft:
* `VERIFIED`: Playbook und Code/Tests stimmen vollständig überein.
* `CONFLICTING`: Nachgewiesener Widerspruch zwischen Playbook-Spezifikation und realem Code.
* `UNSPECIFIED`: Im Code existiert operative Logik oder ein Fallback, der im Playbook nicht definiert ist.
* `NOT_REVIEWED`: Für die Logik liegen keine Tests oder unzureichende Belege vor.
* `NOT_APPLICABLE`: Die Komponente entfällt begründet für diese Strategie.

---

## Phase 4: Standardisierter Audit-Bericht

Der Bericht folgt diesem standardisierten Schema:

```markdown
# Strategy-Audit Bericht: [<strategy_name>]

## 1. Zusammenfassung & Konformitäts-Score
- **Status**: [VOLLSTÄNDIG KONFORM | ABWEICHUNGEN VORHANDEN | KRITISCHE KONFLIKTE]
- **Geprüftes Playbook**: `.agents/skills/strategy-screener/playbook/<strategy_name>.md`
- **Geprüfter Code**: `app/services/screener/strategies/<strategy_name>.py`
- **Übereinstimmungsgrad**: [X] von [Y] Contract-IDs verifiziert

## 2. Detaillierte Matrix-Befunde

| Contract-ID | Bereich | Playbook-Soll | Code-Ist | Status | Schweregrad |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `[STRAT]-IND-01` | Indikatoren | RSI(14) < 30 | RSI(14) < 30 | `VERIFIED` | - |
| `[STRAT]-ENT-02` | Entry-Filter | Close > SMA(200) | Fehlt im Code | `CONFLICTING` | High |

## 3. Identifizierte Abweichungen (Findings)

### [Finding-01] Titel der Abweichung
- **Contract-ID**: `[STRAT]-XXX-YY`
- **Datei/Symbol**: `app/.../strategy.py:Zeile`
- **Klassifizierung**: `CONFLICTING` | `UNSPECIFIED` | `NOT_REVIEWED`
- **Schweregrad**: `Blocking` | `High` | `Medium` | `Low`
- **Evidenz**: Code-Ausschnitt oder fehlende Bedingung.
- **Korrekturvorgabe**: Präziser Vorschlag zur Angleichung (Playbook vs. Code).

## 4. Empfohlene Handlungsschritte
1. [Schritt 1...]
```
