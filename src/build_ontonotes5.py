# src/build_ontonotes5.py
from __future__ import annotations

from typing import Any, Dict, List
from datasets import load_dataset, DatasetDict

from tag_mapping import normalize_bio_sequence


ONTONOTES_ID = "tner/ontonotes5"


def load_ontonotes5(dataset_id: str = ONTONOTES_ID) -> DatasetDict:
    """
    Loads tner/ontonotes5 and returns DatasetDict with:
      - tokens: List[str]
      - tags:   List[str] in CoNLL BIO label space (O, B/I-PER/ORG/LOC/MISC)

    Mapping (collapsed):
      PERSON -> PER
      ORG    -> ORG
      GPE/LOC/FAC -> LOC
      everything else -> MISC
    """
    ds = load_dataset(dataset_id)

    # tner/ontonotes5 uses column name "tags" as int class labels
    names: List[str] = ds["train"].features["tags"].feature.names  # e.g. "B-PERSON"

    def convert(ex: Dict[str, Any]):
        raw = [names[i] for i in ex["tags"]]  # -> strings like "B-WORK_OF_ART"
        tags = normalize_bio_sequence(raw, dataset="ontonotes5")
        return {"tokens": ex["tokens"], "tags": tags}

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        out[split] = ds[split].map(convert, remove_columns=ds[split].column_names)
    return out


if __name__ == "__main__":
    d = load_ontonotes5()
    print({k: len(v) for k, v in d.items()})
    print("Example:", d["train"][0])