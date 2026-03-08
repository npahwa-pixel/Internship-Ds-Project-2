from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: Optional[str]
    display: str

def parse_dataset_spec(s: str) -> DatasetSpec:
    s = s.strip()
    if ":" in s:
        base, cfg = s.split(":", 1)
        return DatasetSpec(base, cfg, s)
    if "@" in s:
        base, cfg = s.split("@", 1)
        return DatasetSpec(base, cfg, s)
    parts = s.split("/")
    if len(parts) >= 3 and parts[0] == "tner":
        base = "/".join(parts[:2])
        cfg = "/".join(parts[2:])
        if base in {"tner/wikiann"} or len(cfg) <= 5:
            return DatasetSpec(base, cfg, s)
    if s == "tner/wikiann":
        return DatasetSpec("tner/wikiann", "en", "tner/wikiann:en")
    return DatasetSpec(s, None, s)
