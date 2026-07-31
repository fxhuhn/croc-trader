---
name: data-ingestion
description: "Data ingestion skill for EOD market data, fallback behavior, provider provenance, synchronization, and data-quality contracts."
---

# Data-Ingestion Skill

This skill defines the data ingestion contract and provider usage rules for the End-of-Day (EOD) trading system.

The architecture documents define the normative system contract. The current implementation, configuration, database schema, and tests are repository evidence that must be inspected before concrete technical details are stated. Do not assume existing behavior is correct when it conflicts with architecture, explicit requirements, or verified contracts.

## Provider Contract

Use `yfinance` as the primary provider for historical daily End-of-Day market
data.

Use TradingView through `tvdatafeed.get_hist()` only when:

- a required symbol is unavailable through `yfinance`, or
- required historical data is missing from the primary provider.

Do not use `TvDatafeedLive`.

Do not introduce:

- live subscriptions,
- streaming bars,
- background listeners,
- intraday processing,
- threaded live consumers,
- asynchronous provider workflows.

Preserve provider selection and fallback diagnostics in the workflow logs, run metadata, or existing repository fields where they are actually supported. Do not claim that every `market_prices` row contains provider provenance: the documented schema has no per-row provider column.

Do not redesign or migrate the provider database schema unless the user explicitly requests a schema change.

## Fallback Semantics

Do not expand the fallback semantics independently. A data quality issue does not automatically trigger a TradingView fallback unless the existing code, tests, or architecture explicitly support it. 

## Technical Details and Evidence Sources

Architecture documents are normative for technical contracts. Implementation, configuration, database schema, and tests provide conformance evidence and concrete technical details, including:
- Provider identifiers and fallback diagnostics where the repository stores them,
- Upsert behavior,
- Database tables,
- Symbol mapping,
- Retry behavior,
- Persistence logic,
- Transaction boundaries.

Do not duplicate these details in this skill.

## Scope Limits

This skill must NOT:
- define new database schema designs,
- add new provenance fields,
- implement live feed logic,
- hardcode concrete credentials or unverified provider parameters,
- invent new operative workflows.

When code changes are required, pass the verified constraints to `python-craftsman`.
