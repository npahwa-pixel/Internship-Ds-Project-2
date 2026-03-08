# src/labels.py
from __future__ import annotations

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
LABEL_LIST = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}