# src/tag_mapping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


CONLL_TYPES = {"PER", "ORG", "LOC", "MISC"}


def _split_bio(tag: str) -> Tuple[str, Optional[str]]:
    """
    Returns (prefix, type) where prefix in {"O","B","I"}.
    Supports tags like:
      - "O"
      - "B-PER", "I-ORG"
      - "B-person" (wnut)
      - "B-WORK_OF_ART" (ontonotes)
    """
    if tag is None:
        return "O", None
    t = str(tag).strip()
    if t == "" or t.upper() == "O":
        return "O", None

    # Standard BIO: PREFIX-TYPE
    if "-" in t:
        p, typ = t.split("-", 1)
        p = p.strip().upper()
        typ = typ.strip()
        if p not in {"B", "I"}:
            # If something odd, treat as entity type with B-
            return "B", t
        return p, typ

    # No prefix, treat as a type
    return "B", t


def _norm_type(typ: str) -> str:
    # Normalize separators and case
    return typ.replace("-", "_").strip().upper()


def map_entity_type(dataset: str, typ: str) -> str:
    """
    Maps a dataset-specific entity type to CoNLL: PER/ORG/LOC/MISC.
    """
    ds = dataset.strip().lower()
    t = _norm_type(typ)

    # WikiANN already is PER/ORG/LOC
    if ds in {"wikiann", "panx", "unimelb_wikiann"}:
        if t in {"PER", "ORG", "LOC"}:
            return t
        return "MISC"

    # tner/ontonotes5: many fine-grained types
    if ds in {"ontonotes", "ontonotes5", "tner_ontonotes5"}:
        if t == "PERSON":
            return "PER"
        if t == "ORG":
            return "ORG"
        if t in {"GPE", "LOC", "FAC"}:
            return "LOC"
        # Everything else -> MISC (DATE, MONEY, NORP, LAW, ...)
        return "MISC"

    # WNUT17: person/location/corporation/group/product/creative-work
    if ds in {"wnut", "wnut17", "wnut_17"}:
        tl = t.lower()
        if tl == "person":
            return "PER"
        if tl == "location":
            return "LOC"
        if tl in {"corporation", "group"}:
            return "ORG"
        if tl in {"product", "creative_work", "creativework", "creative_work"}:
            return "MISC"
        # fallback
        return "MISC"

    # Default fallback
    if t in CONLL_TYPES:
        return t
    return "MISC"


def normalize_bio_sequence(raw_tags: List[str], dataset: str) -> List[str]:
    """
    Converts a tag sequence (BIO-ish) to CoNLL BIO tags.
    Also fixes illegal "I-X" that starts an entity after O / different type -> converts to B-X.
    """
    out: List[str] = []
    prev_type: Optional[str] = None
    prev_inside = False

    for tag in raw_tags:
        p, typ = _split_bio(tag)
        if p == "O" or typ is None:
            out.append("O")
            prev_type = None
            prev_inside = False
            continue

        mapped = map_entity_type(dataset, typ)  # PER/ORG/LOC/MISC

        # BIO validity fix:
        # If I-XXX appears but previous is O or previous type != XXX -> force B-XXX
        if p == "I":
            if (not prev_inside) or (prev_type != mapped):
                p = "B"

        out.append(f"{p}-{mapped}")
        prev_type = mapped
        prev_inside = True

    return out