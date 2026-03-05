from __future__ import annotations

from typing import Any, Dict, List
from datasets import load_dataset, DatasetDict

from tag_mapping import normalize_bio_sequence

WIKIANN_ID = "unimelb-nlp/wikiann"


def load_wikiann(lang: str = "en", dataset_id: str = WIKIANN_ID) -> DatasetDict:
    """
    Returns DatasetDict with:
      - tokens: List[str]
      - tags:   List[str] in CoNLL BIO label space
    """
    ds = load_dataset(dataset_id, lang)

    names: List[str] = ds["train"].features["ner_tags"].feature.names

    def convert(ex: Dict[str, Any]):
        raw = [names[i] for i in ex["ner_tags"]]  # e.g. B-PER
        tags = normalize_bio_sequence(raw, dataset="wikiann")
        return {"tokens": ex["tokens"], "tags": tags}

    out = DatasetDict()
    for split in ["train", "validation", "test"]:
        out[split] = ds[split].map(convert, remove_columns=ds[split].column_names)
    return out


if __name__ == "__main__":
    d = load_wikiann(lang="en")
    print({k: len(v) for k, v in d.items()})
    print("Example:", d["train"][0])