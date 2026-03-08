from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

from nerjson.config.labels import LABEL2ID
from nerjson.data.mapping import map_tag_to_unified


def _load_id2tag_from_hub(ds_name: str) -> Dict[int, str]:
    """T-NER datasets often store label mapping in dataset/label.json."""
    path = hf_hub_download(
        repo_id=ds_name,
        repo_type="dataset",
        filename="dataset/label.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    return {int(v): str(k) for k, v in label2id.items()}


def _try_feature_names(raw_train: Dataset) -> Optional[List[str]]:
    """If tags feature includes names, use them; else fallback to label.json."""
    try:
        feat = raw_train.features["tags"].feature
        names = getattr(feat, "names", None)
        return list(names) if names is not None else None
    except Exception:
        return None


@dataclass
class DatasetBundle:
    name: str
    train: Dataset
    valid: Optional[Dataset]
    test: Optional[Dataset]


def load_and_unify_hf_dataset(ds_name: str, trust_remote_code: bool = False) -> DatasetBundle:
    raw = load_dataset(ds_name, trust_remote_code=trust_remote_code)

    names = _try_feature_names(raw["train"])
    if names is not None:
        id2tag = {i: t for i, t in enumerate(names)}
    else:
        id2tag = _load_id2tag_from_hub(ds_name)

    def convert_split(split: str) -> Dataset:
        def _map(ex):
            if len(ex["tags"]) > 0 and isinstance(ex["tags"][0], str):
                tags_str = ex["tags"]
            else:
                tags_str = [id2tag[int(i)] for i in ex["tags"]]
            unified = [map_tag_to_unified(t) for t in tags_str]
            ex["tags_unified"] = [LABEL2ID[t] for t in unified]
            return ex
        return raw[split].map(_map)

    train = convert_split("train")
    valid = convert_split("validation") if "validation" in raw else None
    test = convert_split("test") if "test" in raw else None
    return DatasetBundle(name=ds_name, train=train, valid=valid, test=test)


def build_multidataset(
    dataset_names: List[str],
    trust_remote_code: bool = False,
) -> Tuple[Dataset, Dataset, Optional[Dataset], List[DatasetBundle]]:
    bundles = [load_and_unify_hf_dataset(n, trust_remote_code=trust_remote_code) for n in dataset_names]

    train_all = concatenate_datasets([b.train for b in bundles])
    valid_all = concatenate_datasets([b.valid for b in bundles if b.valid is not None]) if any(
        b.valid is not None for b in bundles
    ) else None
    test_all = concatenate_datasets([b.test for b in bundles if b.test is not None]) if any(
        b.test is not None for b in bundles
    ) else None

    if valid_all is None:
        raise ValueError("No validation split found across selected datasets.")
    return train_all, valid_all, test_all, bundles
