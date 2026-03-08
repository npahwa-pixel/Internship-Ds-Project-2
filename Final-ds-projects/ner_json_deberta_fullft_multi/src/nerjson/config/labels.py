# Unified BIO tags (PER/ORG/LOC/MISC)
UNIFIED_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

LABEL2ID = {t: i for i, t in enumerate(UNIFIED_TAGS)}
ID2LABEL = {i: t for t, i in LABEL2ID.items()}

# For JSON output
TYPE_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}
