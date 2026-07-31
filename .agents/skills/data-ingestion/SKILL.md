---
name: data-ingestion
description: "Data ingestion skill for EOD market data, fallback behavior, provider provenance, synchronization, and data-quality contracts."
---

# Data-Ingestion Skill

This skill defines the data ingestion contract and provider usage rules for the End-of-Day (EOD) trading system.

The existing implementation and database design are authoritative and correct. This skill documents and protects the existing contract without redesigning it.

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

Preserve the existing provider provenance stored with each database record.

Do not redesign or migrate the existing provider database schema unless the
user explicitly requests a schema change.

## Fallback Semantics

Do not expand the fallback semantics independently. A data quality issue does not automatically trigger a TradingView fallback unless the existing code, tests, or architecture explicitly support it. 

## Technical Details (Authoritative Source)

The existing architecture documents and implementation are authoritative for technical details, including:
- Provider field in the record,
- Provider identifiers,
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
