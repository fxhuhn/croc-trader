---

trigger: path_based
paths:

* "app/routes/api.py"

---

# API Architecture and Design Rules — Mandatory

These rules apply to API endpoints and directly related request or response
contracts in `app/routes/api.py`.

Follow `.agents/AGENTS.md`, `rules/workspace.md`, `rules/python.md`,
`architecture.md`, and `references/architecture.md`.

Do not broaden an API change beyond the explicit task.

## 1. Layer Responsibilities

API routes are transport adapters.

A route may:

* read and validate HTTP input,
* call an existing application service,
* translate the service result into an HTTP response,
* map known application errors to defined status codes,
* record concise operational context.

A route must not:

* implement trading strategy logic,
* calculate indicators,
* contain portfolio or sizing logic,
* execute direct SQL when an appropriate repository or service exists,
* contain provider-specific data-ingestion logic,
* duplicate domain validation,
* place broker orders unless explicitly assigned by architecture.

Keep route functions small and delegate business behavior to services.

## 2. HTTP Methods

Use HTTP methods according to observable behavior.

### GET

Use `GET` only for operations that:

* read data,
* do not modify persistent state,
* do not generate files,
* do not trigger backfills, synchronization, screening, or order generation,
* are safe to repeat.

### POST

Use `POST` for operations that:

* trigger processing,
* start synchronization or backfills,
* create signals, files, or other resources,
* modify database state,
* invalidate or rebuild derived state.

Do not implement state-changing behavior through `GET`.

Use another HTTP method only when an existing verified API contract requires it.

## 3. Canonical Routes

Use lowercase kebab-case for URL path segments.

Use one canonical route for each operation.

Preferred route families are:

* `/screener/<strategy-name>`
* `/trades/backfill`
* `/trades/backfill/<strategy-name>`
* `/orders/generate`
* `/market/sync`
* `/market/reload`

Do not add multiple aliases for the same operation.

Preserve an existing alias only when repository evidence shows that it is a
required compatibility contract. In that case:

* route all aliases to the same application service,
* keep validation and response behavior identical,
* mark the non-canonical alias as deprecated,
* do not add further aliases.

Resolve strategy identifiers through the repository's canonical strategy
mapping. Do not invent strategy names.

## 4. Request Parameters

Use a single canonical name for every parameter.

Do not add convenience abbreviations such as:

* `start` in addition to `start_date`,
* `end` in addition to `end_date`,
* `clear` in addition to `clear_existing`.

Existing aliases may be preserved only for verified backward compatibility
and must be normalized immediately to the canonical name.

### GET Requests

Read query and filtering parameters from `request.args`.

Examples include:

* date filters,
* strategy selection,
* pagination,
* optional read-only flags.

### POST Requests

Use a JSON request body for structured mutation or execution parameters.

Query parameters may be used for a small optional selector only when the
existing API contract requires it.

Do not read the same logical parameter from both the JSON body and the query
string.

Reject unsupported content types where a JSON body is required.

## 5. Input Validation

Validate transport-level input before calling the application service.

Validate:

* required fields,
* field types,
* accepted enumeration values,
* date format and ordering,
* Boolean representation,
* supported strategy identifiers,
* permitted ranges,
* unknown fields when strict schemas are used.

Do not invent defaults for required values.

Defaults are allowed only when defined by configuration, architecture, or an
existing public contract.

Return a client error for invalid input. Do not allow malformed values to reach
domain or persistence layers.

## 6. Response Envelope

Return JSON responses using one of these envelopes.

### Success

```json
{
  "status": "success",
  "result": {}
}
```

### Error

```json
{
  "status": "error",
  "message": "Human-readable error description"
}
```

The `result` value may be an object, list, or scalar when required by the
verified contract.

Do not expose:

* stack traces,
* SQL statements,
* credentials,
* filesystem secrets,
* provider tokens,
* internal exception representations.

Use stable field names.

## 7. HTTP Status Codes

Use status codes according to the verified outcome:

* `200 OK`: successful read or synchronously completed operation.
* `201 Created`: a new identifiable resource was created.
* `202 Accepted`: processing was accepted but is not yet completed.
* `400 Bad Request`: malformed input or missing required transport parameter.
* `403 Forbidden`: the caller is authenticated or identified but not permitted.
* `404 Not Found`: requested resource or canonical strategy does not exist.
* `409 Conflict`: duplicate, overlapping, or conflicting operation.
* `422 Unprocessable Content`: structurally valid input violates a known
  application rule.
* `429 Too Many Requests`: configured request limit exceeded.
* `500 Internal Server Error`: unexpected internal failure.
* `503 Service Unavailable`: required application service is unavailable or not
  initialized.

Do not return `200` for an error response.

Do not use `500` for expected validation, not-found, conflict, or dependency
availability conditions.

## 8. Error Mapping

Catch only expected application exceptions at the route boundary.

Map each known exception to a defined status code and safe error message.

Unexpected exceptions must:

* be logged once with operational context,
* return a generic `500` response,
* preserve the original traceback in server-side logs,
* not reveal sensitive internal details to the caller.

Do not use:

```python
except Exception:
    pass
```

Do not silently convert failures into successful responses.

## 9. Authentication and Authorization

Administrative, trigger, synchronization, backfill, order-generation, or other
state-changing endpoints must use the repository's established access-control
mechanism.

Apply `@require_ip_whitelist` where required by the current architecture.

Do not treat IP allowlisting as proof of complete authentication or
authorization beyond the established system contract.

When proxy headers influence caller identity, trust them only through the
verified deployment configuration.

Do not add or weaken authentication mechanisms without an explicit request and
architecture review.

## 10. Idempotency and Concurrency

State-changing EOD endpoints must preserve the central architecture contracts
for:

* non-overlapping runs,
* effective trading date,
* duplicate protection,
* safe retries,
* atomic persistence,
* order and signal idempotency.

When an operation is already running or has already completed for the same
effective identity, return the status defined by the application contract.

Do not implement duplicate protection independently in each route. Delegate to
the authoritative service or persistence layer.

## 11. Cache Invalidation

Invalidate the Flask view cache only when the completed operation changes data
used by cached views.

Use the established cache abstraction.

Do not silently swallow cache failures.

Catch only known cache exceptions where possible and log the failure with
context.

The endpoint's success behavior must follow the documented application
contract:

* if cache invalidation is required for correctness, treat failure as an
  operation failure;
* if it is a recoverable presentation concern, complete the operation and
  report the cache failure in server-side logs.

Do not use:

```python
try:
    cache.clear()
except Exception:
    pass
```

## 12. Logging

Log operational events at the route or service boundary without duplicating
the same exception at multiple layers.

Include only useful context such as:

* endpoint,
* effective trading date,
* canonical strategy identifier,
* operation identifier,
* result status.

Do not log:

* credentials,
* tokens,
* complete request bodies containing sensitive information,
* unnecessary account information,
* internal secrets.

## 13. Testing

Changes to an API route or its request or response contract require relevant
tests.

Cover as applicable:

* accepted method,
* rejected method,
* valid request,
* missing required parameter,
* invalid type or value,
* unknown strategy,
* access denied,
* service unavailable,
* known application failure,
* unexpected internal failure,
* successful response schema,
* error response schema,
* state-changing idempotency,
* compatibility alias when one must remain.

Do not weaken existing tests to accommodate an implementation.

## 14. Documentation and Architecture Sync

Update `architecture.md` or `references/architecture.md` only when an API change
affects an architecture-relevant contract, including:

* a new API capability,
* a new route family,
* a changed request or response schema,
* a changed access-control boundary,
* a changed external integration,
* a changed state transition or data flow.

Do not maintain a manual inventory of every route function in
`architecture.md`.

Follow the triggers in `architecture-sync`.

## 15. Completion Verification

Before completion:

1. Confirm that state-changing behavior does not use `GET`.
2. Confirm that each parameter has one canonical name.
3. Confirm that input sources match the HTTP method and contract.
4. Confirm that routes delegate business behavior.
5. Confirm that errors use the correct response envelope and status code.
6. Confirm that access control is preserved.
7. Confirm that cache failures are not silently swallowed.
8. Confirm that relevant route tests pass.
9. Confirm that architecture documentation was changed only when required.
10. Report only checks that were actually executed.
