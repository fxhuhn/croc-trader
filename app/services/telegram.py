import html
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Default network timeout in seconds
DEFAULT_TELEGRAM_TIMEOUT_SECONDS: float = 10.0

COLUMN_HEADER_MAPPINGS: dict[str, str] = {
    "symbol": "SYM",
    "sym": "SYM",
    "action": "ACTION",
    "act": "ACTION",
    "signal": "ACTION",
    "entry": "ENTRY",
    "entry_price": "ENTRY",
    "limit entry": "ENTRY",
    "tp": "TP",
    "target": "TP",
    "target_profit": "TP",
    "sl": "SL",
    "stop": "SL",
    "stop_loss": "SL",
    "max close (rsi<40)": "MAX_CL",
    "req_close_rsi40": "MAX_CL",
    "setup close": "CLOSE",
}

OMIT_SECONDARY_COLUMNS: set[str] = {
    "score",
    "setup_score",
    "loc",
    "threshold_loc",
    "atr",
    "atr5",
    "atr%",
    "atr_pct",
    "atr_ratio_3day",
    "rsi",
    "rsi(2)",
    "ibs",
    "volume",
    "sma200",
    "indices",
    "close",
}


MAX_UNFILTERED_COLUMNS: int = 4


def format_compact_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    """Formats headers and string rows into a monospaced ASCII table with matching separator.

    Args:
        headers: List of column header names.
        rows: List of data rows where each row has strings matching headers length.
        alignments: Optional list specifying alignment per column ('left' or 'right').

    Returns:
        str: Monospaced aligned table string.
    """
    if not headers:
        return ""

    num_columns = len(headers)
    resolved_alignments = (
        alignments
        if alignments and len(alignments) == num_columns
        else ["left"] * num_columns
    )

    column_widths = [len(header) for header in headers]
    for row in rows:
        for column_index in range(min(num_columns, len(row))):
            column_widths[column_index] = max(
                column_widths[column_index], len(str(row[column_index]))
            )

    header_cells = []
    for column_index, header in enumerate(headers):
        width = column_widths[column_index]
        if resolved_alignments[column_index] == "right":
            header_cells.append(header.rjust(width))
        else:
            header_cells.append(header.ljust(width))

    spacing = "  "
    header_line = spacing.join(header_cells)
    separator_line = "-" * len(header_line)

    formatted_row_lines = []
    for row in rows:
        row_cells = []
        for column_index in range(num_columns):
            value = str(row[column_index]) if column_index < len(row) else ""
            width = column_widths[column_index]
            if resolved_alignments[column_index] == "right":
                row_cells.append(value.rjust(width))
            else:
                row_cells.append(value.ljust(width))
        formatted_row_lines.append(spacing.join(row_cells))

    return f"{header_line}\n{separator_line}\n" + "\n".join(formatted_row_lines)


def format_dataframe_to_compact_table(dataframe: pd.DataFrame) -> str:
    """Converts a pandas DataFrame into a compact monospaced table string."""
    if dataframe.empty:
        return ""

    available_columns = list(dataframe.columns)
    if len(available_columns) > MAX_UNFILTERED_COLUMNS:
        filtered_columns = [
            column
            for column in available_columns
            if str(column).lower() not in OMIT_SECONDARY_COLUMNS
        ]
        if filtered_columns:
            available_columns = filtered_columns

    headers = [
        COLUMN_HEADER_MAPPINGS.get(str(column).lower(), str(column))
        for column in available_columns
    ]

    alignments = []
    for column in available_columns:
        col_lower = str(column).lower()
        if pd.api.types.is_numeric_dtype(dataframe[column]) or col_lower in {
            "entry",
            "entry_price",
            "limit entry",
            "tp",
            "target",
            "target_profit",
            "sl",
            "stop",
            "stop_loss",
            "close",
            "setup close",
            "max close (rsi<40)",
            "req_close_rsi40",
        }:
            alignments.append("right")
        else:
            alignments.append("left")

    rows: list[list[str]] = []
    for _, row in dataframe.iterrows():
        row_values: list[str] = []
        for column in available_columns:
            value = row[column]
            if pd.isna(value):
                row_values.append("-")
            elif isinstance(value, float):
                row_values.append(f"{value:.2f}")
            else:
                row_values.append(str(value))
        rows.append(row_values)

    return format_compact_table(headers, rows, alignments)


class TelegramBot:
    """Dispatches operational alerts and daily market data tables to Telegram.

    Acts as an external integration gateway within the imperative shell layer,
    ensuring sensitive API tokens are sanitized prior to log recording.

    Attributes:
        bot_token: Secret Telegram API authentication token.
        chat_id: Target Telegram channel or user chat identifier.
        enabled: Controls whether outbound notifications are active.
    """

    bot_token: str
    chat_id: str
    enabled: bool

    def __init__(self, token: str, chat_id: str, enabled: bool = False) -> None:
        """Initializes the Telegram bot client context.

        Args:
            token: Telegram bot token.
            chat_id: Destination chat identifier.
            enabled: Flags if sending messages is permitted.
        """
        self.bot_token = token
        self.chat_id = chat_id
        self.enabled = enabled

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot not fully configured (token/chat_id missing).")

    def send(self, text: str, parse_mode: str = "Markdown") -> dict[str, object] | None:
        """Legacy compatibility wrapper for message dispatching.

        Args:
            text: Message body to transmit.
            parse_mode: Formatting engine designation.

        Returns:
            dict[str, object] | None: Parsed JSON response payload from Telegram.
        """
        logger.debug("Executing legacy Telegram send wrapper.")
        return self.send_message(text, parse_mode)

    def send_message(
        self, text: str, parse_mode: str = "Markdown"
    ) -> dict[str, object] | None:
        """Transmits a formatted text message to the configured Telegram chat.

        Args:
            text: Raw message content.
            parse_mode: Telegram markup mode ('Markdown' or 'HTML').

        Returns:
            dict[str, object] | None: API response payload or None on failure/skip.
        """
        if not self.enabled:
            logger.debug("Telegram Send skipped (disabled).")
            return None

        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot not configured.")
            return None

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload: dict[str, str | bool] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(
                url, json=payload, timeout=DEFAULT_TELEGRAM_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            json_payload: dict[str, object] = response.json()
            return json_payload
        except requests.exceptions.RequestException as network_error:
            error_message = str(network_error)
            if self.bot_token:
                error_message = error_message.replace(self.bot_token, "********")
            logger.error("Telegram network error: %s", error_message)
            return None

    def send_dataframe(
        self, dataframe: pd.DataFrame, title: str = ""
    ) -> dict[str, object] | None:
        """Formats a pandas DataFrame into a monospaced ASCII table in Telegram HTML format.

        Args:
            dataframe: Tabular data to be displayed.
            title: Header title preceding the table.

        Returns:
            dict[str, object] | None: API response payload or None if skipped.
        """
        if not self.enabled:
            return None

        escaped_title = html.escape(title)

        if dataframe.empty:
            message = (
                f"<b>{escaped_title}</b>\n<i>(No Data)</i>"
                if escaped_title
                else "<i>(No Data)</i>"
            )
            return self.send_message(message, parse_mode="HTML")

        table_string = format_dataframe_to_compact_table(dataframe)
        message = (
            f"<b>{escaped_title}</b>\n<pre>{table_string}</pre>"
            if escaped_title
            else f"<pre>{table_string}</pre>"
        )
        return self.send_message(message, parse_mode="HTML")
