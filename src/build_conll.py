# src/build_conll.py
from __future__ import annotations
from typing import List, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict

def load_conll() -> DatasetDict:
    """
    Uses HuggingFace 'conll2003'. It provides:
      - tokens: List[str]
      - ner_tags: List[int] with ClassLabel names like B-PER, I-ORG, O, ...
    We will convert ints -> string labels and keep tokens + tags.
    """
    ds = load_dataset("conll2003")

    # map int labels to names
    names: List[str] = ds["train"].features["ner_tags"].feature.names  # ClassLabel names

    def convert(ex: Dict[str, Any]):
        tags = [names[i] for i in ex["ner_tags"]]  # e.g. "B-PER"
        # CoNLL already uses PER/ORG/LOC/MISC so no mapping needed
        return {"tokens": ex["tokens"], "tags": tags}

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        keep = ds[split].map(convert, remove_columns=[c for c in ds[split].column_names if c not in ("tokens", "ner_tags")])
        out[split] = keep
    return out

if __name__ == "__main__":
    d = load_conll()
    print({k: len(v) for k, v in d.items()})
    print("Example:", d["train"][0])