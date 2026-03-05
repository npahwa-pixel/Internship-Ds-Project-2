# src/build_external_datasets.py
from __future__ import annotations

from datasets import DatasetDict

from build_wikiann import load_wikiann
from build_ontonotes5 import load_ontonotes5
from build_wnut17 import load_wnut17
from build_conll import load_conll


def load_named_dataset(name: str, lang: str = "en") -> DatasetDict:
    """
    name in: conll, wikiann, ontonotes5, wnut17
    """
    n = name.strip().lower()
    if n in {"conll", "conll2003"}:
        return load_conll()
    if n in {"wikiann", "panx"}:
        return load_wikiann(lang=lang)
    if n in {"ontonotes", "ontonotes5"}:
        return load_ontonotes5()
    if n in {"wnut", "wnut17", "wnut_17"}:
        return load_wnut17()
    raise ValueError(f"Unknown dataset name: {name}")