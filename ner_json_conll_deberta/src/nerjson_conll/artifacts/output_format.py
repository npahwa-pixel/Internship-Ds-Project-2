from __future__ import annotations

import json
from typing import Any, Dict, List


def as_json(entities: List[Dict[str, Any]]) -> str:
    return json.dumps({"entities": entities}, ensure_ascii=False)


def as_plain(entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return "(no entities)"
    lines = []
    for e in entities:
        lines.append(f"{e.get('type')}: {e.get('text')} [{e.get('start')},{e.get('end')}]")
    return "\n".join(lines)


def as_markdown(entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return "_No entities_"
    header = "| type | text | start | end |\n|---|---|---:|---:|"
    rows: List[str] = []
    for e in entities:
        txt = str(e.get("text", ""))
        txt = txt.replace("|", "\\|")  # escape pipes for markdown tables (safe in py3.9)
        rows.append(f"| {e.get('type')} | {txt} | {e.get('start')} | {e.get('end')} |")
    return "\n".join([header] + rows)


def as_xml(entities: List[Dict[str, Any]]) -> str:
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    parts = ["<entities>"]
    for e in entities:
        parts.append(
            f'  <entity type="{esc(str(e.get("type", "")))}" start="{int(e.get("start", 0))}" end="{int(e.get("end", 0))}">{esc(str(e.get("text", "")))}</entity>'
        )
    parts.append("</entities>")
    return "\n".join(parts)


def format_output(fmt: str, entities: List[Dict[str, Any]]) -> str:
    f = (fmt or "json").strip().lower()
    if f in {"json", "application/json"}:
        return as_json(entities)
    if f in {"plain", "text", "txt"}:
        return as_plain(entities)
    if f in {"markdown", "md"}:
        return as_markdown(entities)
    if f == "xml":
        return as_xml(entities)
    return as_json(entities)
