from __future__ import annotations

def normalize_ent_type(ent: str) -> str:
    x = ent.strip().lower()
    if x in {"per", "person"}:
        return "PER"
    if x in {"org", "organization", "corporation", "company", "group", "agency"}:
        return "ORG"
    if x in {"loc", "location", "gpe", "geo-loc", "geoloc", "facility", "fac", "place"}:
        return "LOC"
    return "MISC"


def map_tag_to_unified(tag: str) -> str:
    tag = str(tag)
    if tag == "O":
        return "O"
    if "-" not in tag:
        return "O"
    prefix, ent = tag.split("-", 1)
    prefix = prefix.upper()
    if prefix not in {"B", "I"}:
        return "O"
    return f"{prefix}-{normalize_ent_type(ent)}"
