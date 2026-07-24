# API Architecture & Design Rules — MANDATORY

> [!CAUTION]
> These rules are NON-NEGOTIABLE for all API endpoints in Croc-Trader (`app/routes/api.py`).
> All new and refactored API routes must comply with these guidelines.

---

## 1. HTTP Methods & REST Semantics

- **Strict Method Separation**:
  - **`POST` for Side Effects & Mutations**: Any endpoint that triggers a process, runs a backfill, modifies SQLite database state, or generates files MUST strictly use `methods=["POST"]`. `GET` is FORBIDDEN for state-changing operations.
  - **`GET` for Pure Reads & Queries**: `GET` is reserved exclusively for idempotent read operations (status checks, metric queries, data retrieval).
- **HTTP Status Codes**:
  - `200 OK`: Successful read or execution response.
  - `201 Created`: Resource successfully generated (e.g., order CSV files).
  - `400 Bad Request`: Invalid or missing required parameters.
  - `403 Forbidden`: Unauthorized IP access.
  - `404 Not Found`: Unknown endpoint or unmapped strategy.
  - `500 Internal Server Error`: Unhandled backend exception.
  - `503 Service Unavailable`: Uninitialized engine or service dependency.

---

## 2. URL Path Conventions

- **Kebab-Case Naming**: URL path segments MUST use kebab-case (`dip-buyer`, `ndx-momentum`, `tgim`).
- **Standard Hierarchy**:
  - **Screener Endpoints**: `/screener/<strategy-name>` (e.g., `/screener/dip-buyer`, `/screener/tgim`)
  - **Trade & Backfill Endpoints**: `/trades/backfill`, `/trades/backfill/<strategy-name>`
  - **Order Operations**: `/orders/generate`
  - **Market Operations**: `/market/sync`, `/market/reload`
- **Route Aliasing**: Provide both generic strategy-parameterized routes (`/trades/backfill?strategy=tgim`) and dedicated strategy routes (`/trades/backfill/tgim`).

---

## 3. Parameter Resolution Standard

- **Exclusive URL Query Parameters**: API endpoints MUST extract parameters EXCLUSIVELY from **URL Query Parameters** (`request.args`). JSON body payloads (`request.get_json()`) are FORBIDDEN for API configuration parameters.
- **Resolution Flow**:
  1. Primary Query Parameter (`request.args.get("start_date")`)
  2. Alias Query Parameter (`request.args.get("start")`)
  3. Predefined Fallback Default
- **Parameter Aliasing**: Support both full names and common abbreviations (e.g., `start_date` / `start`, `end_date` / `end`, `clear_existing` / `clear`).

---

## 4. Uniform JSON Response Schema

All API responses MUST follow a standardized JSON envelope structure:

### Success Response
```json
{
  "status": "success",
  "result": {
    "start_date": "2026-01-01",
    "total_pnl": 567.37,
    "trades_closed": 6
  }
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Detailed human-readable error description"
}
```

---

## 5. Security & Cache Invalidation

- **IP Whitelisting**: Every administrative, trigger, or backfill API route MUST be protected with `@require_ip_whitelist`.
- **Cache Clearing**: Endpoints modifying database state MUST safely invalidate the Flask view cache (`try: cache.clear() except Exception: pass`) to ensure immediate web UI updates.
- **Architecture Sync**: All public API functions MUST be cataloged in `architecture.md` Section 4 to maintain 100% compliance with `architecture-sync-check`.
