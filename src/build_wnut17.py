# src/build_wnut17.py
from __future__ import annotations

from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict

from tag_mapping import normalize_bio_sequence


WNUT_ID = "leondz/wnut_17"


def load_wnut17(dataset_id: str = WNUT_ID) -> DatasetDict:
    """
    Loads WNUT17 and returns DatasetDict with:
      - tokens: List[str]
      - tags:   List[str] in CoNLL BIO label space (O, B/I-PER/ORG/LOC/MISC)

    WNUT types:
      person, location, corporation, group, product, creative-work
    Collapsed mapping:
      person -> PER
      location -> LOC
      corporation/group -> ORG
      product/creative-work -> MISC
    """
    ds = load_dataset(dataset_id)

    # WNUT is stored as ClassLabel; values might come as ints (common) or strings (depending on builder)
    names: List[str] = ds["train"].features["ner_tags"].feature.names

    def to_str_tags(seq: List[Union[int, str]]) -> List[str]:
        out: List[str] = []
        for x in seq:
            if isinstance(x, int):
                out.append(names[x])
            else:
                out.append(str(x))
        return out

    def convert(ex: Dict[str, Any]):
        raw = to_str_tags(ex["ner_tags"])  # e.g. "B-person"
        tags = normalize_bio_sequence(raw, dataset="wnut17")
        return {"tokens": ex["tokens"], "tags": tags}

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        out[split] = ds[split].map(convert, remove_columns=ds[split].column_names)
    return out


if __name__ == "__main__":
    d = load_wnut17()
    print({k: len(v) for k, v in d.items()})
    print("Example:", d["train"][0])