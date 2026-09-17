"""Static linting and continuous assurance suite for Jinja2 templates.

Validates typography hierarchies, font weights, ad-hoc pixel classes,
monospaced font usage, section header macro invocations, mandatory
semantic <h1> elements, Jinja2 syntax correctness, and German locale
number formatting (de-DE: 1.234,56 $).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import jinja2
import pytest

# Maximum allowed legacy violations per template file to enforce non-regression.
# Decrement these limits whenever existing templates are migrated and cleaned up.
LEGACY_TECHNICAL_DEBT_BASELINE: Final[dict[str, int]] = {}

FINANCIAL_METRIC_KEYWORDS: Final[tuple[str, ...]] = (
    "price",
    "pnl",
    "score",
    "atr",
    "close",
    "roc",
    "ev_",
    "sharpe",
    "sortino",
    "trades_per_month",
    "val-weight",
)

MANDATORY_H1_TEMPLATES: Final[frozenset[str]] = frozenset(
    {
        "404.html",
        "500.html",
        "honeypot_login.html",
        "analytics.html",
        "analytics_monthly_matrix.html",
        "backtest_dashboard.html",
        "trades.html",
        "trades_base.html",
        "trades_broker.html",
        "screener.html",
        "screener_base.html",
    }
)


@dataclass(frozen=True)
class LintViolation:
    """Immutable representation of a template linting violation."""

    rule_name: str
    line_number: int
    snippet: str


def _check_jinja_syntax(content: str) -> list[LintViolation]:
    """Compiles Jinja2 template AST to catch syntax or parsing errors."""
    env = jinja2.Environment()
    try:
        env.parse(content)
        return []
    except jinja2.TemplateSyntaxError as error:
        return [
            LintViolation(
                "JINJA_SYNTAX_ERROR",
                error.lineno or 1,
                f"Jinja2 syntax error: {error.message}",
            )
        ]


def _check_mandatory_h1(content: str, relative_path: str) -> list[LintViolation]:
    """Ensures that all primary views and base page layouts define exactly one semantic <h1> element."""
    normalized_path = relative_path.replace("\\", "/")
    if normalized_path not in MANDATORY_H1_TEMPLATES:
        return []

    h1_matches = re.findall(r"<h1\b", content, re.IGNORECASE)
    if not h1_matches:
        return [
            LintViolation(
                "MISSING_SEMANTIC_H1",
                1,
                f"Template '{normalized_path}' is a primary view/layout but lacks a semantic <h1> header.",
            )
        ]
    if len(h1_matches) > 1:
        return [
            LintViolation(
                "MULTIPLE_H1_ELEMENTS",
                1,
                f"Template '{normalized_path}' contains {len(h1_matches)} <h1> elements (expected exactly 1).",
            )
        ]
    return []


def _check_locale_violations(line: str, line_number: int) -> list[LintViolation]:
    """Detects US-style raw formatting and prefix currency symbols."""
    violations: list[LintViolation] = []
    has_us_format = bool(
        re.search(r"\{:[\+\-\s]*,\.?\d*f\}", line)
        or re.search(r'"%[\.\d]*f"\|format', line)
    )
    if has_us_format:
        violations.append(LintViolation("US_RAW_FORMAT", line_number, line.strip()))

    clean_line = re.sub(r"\$\{[a-zA-Z0-9_\.]+\}", "", line)
    has_currency_prefix = bool(re.search(r"\$\s*(?:\{\{|\d|<span)", clean_line))
    if has_currency_prefix:
        violations.append(
            LintViolation("US_CURRENCY_PREFIX", line_number, line.strip())
        )
    return violations


def _check_font_weight_violations(line: str, line_number: int) -> list[LintViolation]:
    """Detects forbidden font weight utility classes (strictly enforces 3-tier system)."""
    pattern = r"\b(font-thin|font-extralight|font-light|font-semibold|font-extrabold|font-black)\b"
    matches = re.findall(pattern, line)
    if matches:
        return [
            LintViolation(
                "FORBIDDEN_FONT_WEIGHT",
                line_number,
                f"Prohibited font weight '{match}': {line.strip()}",
            )
            for match in matches
        ]
    return []


def _check_adhoc_size_violations(line: str, line_number: int) -> list[LintViolation]:
    """Detects ad-hoc arbitrary pixel font size classes."""
    if re.search(r"\btext-\[\d+px\]", line):
        return [LintViolation("ADHOC_PIXEL_SIZE", line_number, line.strip())]
    return []


def _check_font_mono_violations(line: str, line_number: int) -> list[LintViolation]:
    """Detects misapplied font-mono classes on financial metrics and prices."""
    if "font-mono" not in line:
        return []
    lowered = line.lower()
    if any(keyword in lowered for keyword in FINANCIAL_METRIC_KEYWORDS):
        return [
            LintViolation(
                "FONT_MONO_ON_FINANCIAL_DATA",
                line_number,
                line.strip(),
            )
        ]
    return []


def _check_inline_header_violations(
    line: str, line_number: int, is_macro_file: bool
) -> list[LintViolation]:
    """Detects inline h2 tags that bypass the render_section_header macro."""
    if is_macro_file:
        return []
    if "<h2" in line.lower():
        return [
            LintViolation(
                "INLINE_H2_WITHOUT_MACRO",
                line_number,
                line.strip(),
            )
        ]
    return []


def _check_standalone_font_imports(
    line: str, line_number: int, is_base_layout: bool
) -> list[LintViolation]:
    """Detects standalone @import font declarations in templates."""
    if is_base_layout:
        return []
    if "@import" in line and "fonts.googleapis" in line:
        return [
            LintViolation(
                "STANDALONE_FONT_IMPORT",
                line_number,
                line.strip(),
            )
        ]
    return []


def scan_template_content(
    content: str,
    relative_path: str = "template.html",
) -> list[LintViolation]:
    """Scans template content and returns all detected design token violations.

    Args:
        content: Jinja2 template raw text content.
        relative_path: Repository-relative template file path.

    Returns:
        List of immutable LintViolation records.
    """
    violations: list[LintViolation] = []
    violations.extend(_check_jinja_syntax(content))
    violations.extend(_check_mandatory_h1(content, relative_path))

    lines = content.splitlines()
    is_macro_file = relative_path.endswith("macros/cards.html")
    is_base_layout = relative_path.endswith("base.html")

    for line_index, line in enumerate(lines, start=1):
        violations.extend(_check_locale_violations(line, line_index))
        violations.extend(_check_font_weight_violations(line, line_index))
        violations.extend(_check_adhoc_size_violations(line, line_index))
        violations.extend(_check_font_mono_violations(line, line_index))
        violations.extend(
            _check_inline_header_violations(line, line_index, is_macro_file)
        )
        violations.extend(
            _check_standalone_font_imports(line, line_index, is_base_layout)
        )

    return violations


# ---------------------------------------------------------------------------
# Unit Tests for Linting Rules
# ---------------------------------------------------------------------------


def test_lint_detects_jinja_syntax_errors() -> None:
    """Verifies detection of unclosed parentheses or malformed Jinja2 syntax."""
    valid_snippet = "{% set x = (1 if a else 2) %}"
    invalid_snippet = "{% set x = (1 if a else 2)) %}"

    assert scan_template_content(valid_snippet) == []
    violations = scan_template_content(invalid_snippet)
    assert len(violations) == 1
    assert violations[0].rule_name == "JINJA_SYNTAX_ERROR"


def test_lint_detects_forbidden_font_weights() -> None:
    """Verifies detection of font-light, font-black, and font-semibold weights."""
    valid_snippet = '<p class="font-normal text-slate-600 font-bold">Valid</p>'
    invalid_snippet = '<h1 class="font-light text-3xl font-black"><span class="font-semibold">Title</span></h1>'

    assert scan_template_content(valid_snippet) == []
    violations = scan_template_content(invalid_snippet)
    assert len(violations) == 3
    assert all(v.rule_name == "FORBIDDEN_FONT_WEIGHT" for v in violations)


def test_lint_detects_mandatory_h1_structure() -> None:
    """Verifies that primary view templates must contain exactly one <h1> element."""
    valid_page = """{% extends "base.html" %}
    <main>
        <h1 class="text-2xl font-bold text-slate-900">Dashboard</h1>
    </main>"""
    missing_h1_page = """{% extends "base.html" %}
    <main>
        <h2 class="text-xl font-bold text-slate-900">Dashboard</h2>
    </main>"""
    multiple_h1_page = """{% extends "base.html" %}
    <main>
        <h1 class="text-2xl font-bold text-slate-900">Title 1</h1>
        <h1 class="text-2xl font-bold text-slate-900">Title 2</h1>
    </main>"""

    assert scan_template_content(valid_page, "analytics.html") == []
    missing_violations = scan_template_content(missing_h1_page, "analytics.html")
    assert any(v.rule_name == "MISSING_SEMANTIC_H1" for v in missing_violations)

    multiple_violations = scan_template_content(multiple_h1_page, "analytics.html")
    assert any(v.rule_name == "MULTIPLE_H1_ELEMENTS" for v in multiple_violations)


def test_lint_detects_adhoc_pixel_sizes() -> None:
    """Verifies detection of arbitrary text-[...px] classes."""
    valid_snippet = '<span class="text-xs text-slate-500">Subline</span>'
    invalid_snippet = '<span class="text-[11px] text-slate-500">Badge</span>'

    assert scan_template_content(valid_snippet) == []
    violations = scan_template_content(invalid_snippet)
    assert len(violations) == 1
    assert violations[0].rule_name == "ADHOC_PIXEL_SIZE"


def test_lint_detects_locale_violations() -> None:
    """Verifies detection of US raw format strings and currency prefixes."""
    valid_snippet = "<span>{{ price | de_number(2) }}&nbsp;$</span>"
    invalid_snippet_format = '<span>{{ "{:,.2f}".format(price) }}</span>'
    invalid_snippet_currency = "<span>${{ price }}</span>"

    assert scan_template_content(valid_snippet) == []
    violations_format = scan_template_content(invalid_snippet_format)
    assert any(item.rule_name == "US_RAW_FORMAT" for item in violations_format)

    violations_currency = scan_template_content(invalid_snippet_currency)
    assert any(item.rule_name == "US_CURRENCY_PREFIX" for item in violations_currency)


def test_lint_detects_font_mono_on_financial_metrics() -> None:
    """Verifies detection of font-mono on metrics, while allowing it on IDs."""
    valid_id_snippet = '<span class="font-mono">{{ order.order_id }}</span>'
    invalid_metric_snippet = (
        '<span class="font-mono font-bold">{{ row.close_price }}</span>'
    )

    assert scan_template_content(valid_id_snippet) == []
    violations = scan_template_content(invalid_metric_snippet)
    assert len(violations) == 1
    assert violations[0].rule_name == "FONT_MONO_ON_FINANCIAL_DATA"


def test_lint_detects_inline_h2_headers() -> None:
    """Verifies that inline h2 tags are flagged outside macro definitions."""
    valid_macro_call = '{{ render_section_header("Positions", icon="zap") }}'
    invalid_inline_h2 = '<h2 class="text-lg font-bold">Positions</h2>'

    assert scan_template_content(valid_macro_call) == []
    violations = scan_template_content(invalid_inline_h2)
    assert len(violations) == 1
    assert violations[0].rule_name == "INLINE_H2_WITHOUT_MACRO"


def test_lint_allows_h2_inside_cards_macro_definition() -> None:
    """Verifies that the canonical h2 declaration in cards.html is permitted."""
    macro_def = '<h2 class="text-xs md:text-lg font-bold">{{ title }}</h2>'
    assert scan_template_content(macro_def, "macros/cards.html") == []


# ---------------------------------------------------------------------------
# Repository Non-Regression Gate
# ---------------------------------------------------------------------------


def _collect_template_paths() -> list[Path]:
    """Collects all HTML template files within app/templates."""
    workspace_root = Path(__file__).resolve().parents[3]
    templates_dir = workspace_root / "app" / "templates"
    return sorted(templates_dir.rglob("*.html"))


@pytest.mark.parametrize(
    "template_path",
    _collect_template_paths(),
    ids=lambda path: str(path.name),
)
def test_template_typography_and_locale_non_regression(
    template_path: Path,
) -> None:
    """Asserts that templates do not exceed their registered technical debt baseline.

    New templates must pass cleanly with 0 violations. Existing templates
    are constrained to their baseline debt limit and cannot regress.
    """
    workspace_root = Path(__file__).resolve().parents[3]
    rel_path = str(template_path.relative_to(workspace_root / "app" / "templates"))
    content = template_path.read_text(encoding="utf-8")
    violations = scan_template_content(content, relative_path=rel_path)

    allowed_baseline = LEGACY_TECHNICAL_DEBT_BASELINE.get(rel_path, 0)
    actual_violations = len(violations)

    failure_details = "\n".join(
        f"  L{viol.line_number} [{viol.rule_name}]: {viol.snippet}"
        for viol in violations
    )

    assert actual_violations <= allowed_baseline, (
        f"Template '{rel_path}' introduced {actual_violations - allowed_baseline} "
        f"new typography/locale violation(s) (Limit: {allowed_baseline}, Actual: {actual_violations}):\n"
        f"{failure_details}"
    )
