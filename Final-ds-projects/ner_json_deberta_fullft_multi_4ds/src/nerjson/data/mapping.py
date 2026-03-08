from __future__ import annotations
from nerjson.config.labels import LABEL2ID

def normalize_ent_type(ent: str) -> str:
    x = str(ent).strip().lower()
    if x in {"per", "person"}:
        return "PER"
    if x in {"org", "organization", "organisation", "corporation", "company", "agency"}:
        return "ORG"
    if x in {
        "loc", "location", "gpe", "facility", "fac", "place",
        "geo-loc", "geoloc",
        "geopolitical_area", "geopolitical", "geopoliticalarea",
        "geographical_area", "geographical",
    }:
        return "LOC"
    if x in {"group"}:
        return "ORG"
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
