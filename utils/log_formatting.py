"""Shared helpers for readable multi-line log panels."""

import sys
from typing import Any, Iterable, Tuple

_PLAIN_BLOCK_STYLES = {
    "default": {"line_char": "-", "label_prefix": "", "label_suffix": ""},
    "interaction": {"line_char": "=", "label_prefix": "[[ ", "label_suffix": " ]]"},
    "dispatch": {"line_char": "~", "label_prefix": "{{ ", "label_suffix": " }}"},
    "llm": {"line_char": "#", "label_prefix": "## ", "label_suffix": " ##"},
}


def format_log_panel(
    title: str,
    fields: Iterable[Tuple[str, Any]],
    *,
    panel_inner_width: int = 98,
    max_value_chars: int = 220,
) -> str:
    """
    Format key/value pairs into a compact rounded panel for log readability.

    Example output:
    ╭───────────────────────────── INTERACTION HEADER ─────────────────────────────╮
    │ user        : alice (123)                                                    │
    │ channel_id  : 456                                                            │
    ╰───────────────────────────────────────────────────────────────────────────────╯
    """

    normalized: list[tuple[str, str]] = []
    for key, value in fields:
        key_text = str(key).strip() or "field"
        value_text = str(value).replace("\n", "\\n").strip()
        if len(value_text) > max_value_chars:
            value_text = value_text[: max_value_chars - 3] + "..."
        normalized.append((key_text, value_text))

    if not normalized:
        normalized = [("info", "n/a")]

    key_width = min(24, max(len(key) for key, _ in normalized))

    raw_lines: list[str] = []
    for key, value in normalized:
        raw_lines.append(f"{key.ljust(key_width)} : {value}")

    title_text = f"  {title.strip() or 'SUMMARY'}  "
    content_width = max(len(title_text) + 2, max(len(line) for line in raw_lines))
    content_width = min(content_width, max(24, panel_inner_width))

    clipped_title = title_text
    if len(clipped_title) > content_width:
        clipped_title = clipped_title[: max(1, content_width - 1)] + "…"

    top_fill = max(0, content_width - len(clipped_title))
    top_left = top_fill // 2
    top_right = top_fill - top_left
    top_border = "╭" + ("─" * top_left) + clipped_title + ("─" * top_right) + "╮"
    bottom_border = "╰" + ("─" * content_width) + "╯"

    lines = [top_border]
    for line in raw_lines:
        clipped = line[:content_width]
        lines.append("│" + clipped.ljust(content_width) + "│")
    lines.append(bottom_border)
    return "\n".join(lines)


def emit_plain_block_marker(label: str, *, width: int = 119, style: str = "default") -> None:
    """
    Emit a plain separator block directly to stdout.
    This bypasses logger format prefixes for cleaner visual boundaries.
    """
    style_config = _PLAIN_BLOCK_STYLES.get(style, _PLAIN_BLOCK_STYLES["default"])
    line_char = str(style_config.get("line_char", "-"))[:1] or "-"
    label_prefix = str(style_config.get("label_prefix", ""))
    label_suffix = str(style_config.get("label_suffix", ""))
    line = line_char * max(32, width)
    marker = (label or "").strip() or "BLOCK"
    decorated_marker = f"{label_prefix}{marker}{label_suffix}"
    sys.stdout.write(f"{line}\n{decorated_marker}\n{line}\n")
    sys.stdout.flush()
