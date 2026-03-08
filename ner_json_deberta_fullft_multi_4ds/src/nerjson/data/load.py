from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

from nerjson.config.labels import LABEL2ID
from nerjson.data.mapping import map_tag_to_unified
from nerjson.data.spec import DatasetSpec


CANON_FEATURES = Features(
    {"tokens": Sequence(Value("string")), "tags_unified": Sequence(Value("int32"))}
)

def _ensure_tokens_tags_columns(ds: Dataset) -> Dataset:
    cols = set(ds.column_names)
    if "tokens" not in cols:
        if "words" in cols:
            ds = ds.rename_column("words", "tokens")
        else:
            raise RuntimeError(f"Could not find tokens/words column. columns={ds.column_names}")
    if "tags" not in cols:
        if "ner_tags" in cols:
            ds = ds.rename_column("ner_tags", "tags")
        elif "labels" in cols:
            ds = ds.rename_column("labels", "tags")
        else:
            raise RuntimeError(f"Could not find tags/ner_tags/labels column. columns={ds.column_names}")
    return ds

def _try_feature_names(raw_train: Dataset) -> Optional[List[str]]:
    try:
        feats = raw_train.features
        for col in ("tags", "ner_tags", "labels"):
            if col not in feats:
                continue
            feat = feats[col]
            inner = getattr(feat, "feature", feat)
            names = getattr(inner, "names", None)
            if names is not None:
                return list(names)
    except Exception:
        pass
    return None

def _parse_label_mapping_json(data: dict) -> Dict[int, str]:
    if not isinstance(data, dict):
        raise ValueError("label mapping JSON must be an object")
    if "id2label" in data and isinstance(data["id2label"], dict):
        data = data["id2label"]
    elif "id2tag" in data and isinstance(data["id2tag"], dict):
        data = data["id2tag"]
    elif "label2id" in data and isinstance(data["label2id"], dict):
        data = data["label2id"]
    if all(isinstance(k, str) and k.isdigit() for k in data.keys()):
        return {int(k): str(v) for k, v in data.items()}
    if all(isinstance(v, (int, float)) for v in data.values()):
        return {int(v): str(k) for k, v in data.items()}
    raise ValueError("Unrecognized label mapping JSON format")

def _load_id2tag_from_hub(ds_repo_id: str) -> Dict[int, str]:
    candidates = ("dataset/label.json", "label.json", "dataset/labels.json", "labels.json")
    last_err = None
    for repo_type in ("dataset", None):
        for filename in candidates:
            try:
                if repo_type is None:
                    local = hf_hub_download(repo_id=ds_repo_id, filename=filename)
                else:
                    local = hf_hub_download(repo_id=ds_repo_id, repo_type=repo_type, filename=filename)
                with open(local, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return _parse_label_mapping_json(data)
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not load label mapping for dataset={ds_repo_id}. Last error: {last_err}")

@dataclass
class DatasetBundle:
    spec: DatasetSpec
    train: Dataset
    valid: Optional[Dataset]
    test: Optional[Dataset]

def _should_trust_remote_code(spec: DatasetSpec, trust_remote_code_flag: bool) -> bool:
    return True if spec.name == "tner/wikiann" else trust_remote_code_flag

def load_and_unify_hf_dataset(spec: DatasetSpec, trust_remote_code_flag: bool) -> DatasetBundle:
    trust = _should_trust_remote_code(spec, trust_remote_code_flag)
    raw = load_dataset(spec.name, spec.config, trust_remote_code=trust) if spec.config else load_dataset(spec.name, trust_remote_code=trust)
    for split in list(raw.keys()):
        raw[split] = _ensure_tokens_tags_columns(raw[split])
    names = _try_feature_names(raw["train"])
    if names is not None:
        id2tag = {i: t for i, t in enumerate(names)}
    else:
        id2tag = _load_id2tag_from_hub(spec.name)

    def convert_split(split: str) -> Dataset:
        def _map(ex):
            if len(ex["tags"]) > 0 and isinstance(ex["tags"][0], str):
                tags_str = ex["tags"]
            else:
                tags_str = [id2tag[int(i)] for i in ex["tags"]]
            unified = [map_tag_to_unified(t) for t in tags_str]
            ex["tags_unified"] = [np.int32(LABEL2ID[t]).item() for t in unified]
            return ex
        ds = raw[split].map(_map)
        keep = {"tokens", "tags_unified"}
        drop_cols = [c for c in ds.column_names if c not in keep]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        return ds.cast(CANON_FEATURES)

    train = convert_split("train")
    valid = convert_split("validation") if "validation" in raw else None
    test = convert_split("test") if "test" in raw else None
    return DatasetBundle(spec=spec, train=train, valid=valid, test=test)

def build_multidataset(dataset_specs: List[DatasetSpec], trust_remote_code_flag: bool) -> Tuple[Dataset, Dataset, Optional[Dataset], List[DatasetBundle]]:
    bundles = [load_and_unify_hf_dataset(s, trust_remote_code_flag) for s in dataset_specs]
    train_all = concatenate_datasets([b.train for b in bundles])
    valid_all = concatenate_datasets([b.valid for b in bundles if b.valid is not None]) if any(b.valid is not None for b in bundles) else None
    test_all = concatenate_datasets([b.test for b in bundles if b.test is not None]) if any(b.test is not None for b in bundles) else None
    if valid_all is None:
        raise ValueError("No validation split found across selected datasets.")
    return train_all, valid_all, test_all, bundles
