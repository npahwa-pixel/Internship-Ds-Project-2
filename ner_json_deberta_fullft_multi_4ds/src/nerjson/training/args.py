from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class TrainConfig:
    model: str
    datasets: List[str]
    output_dir: str = "models/deberta_fullft_multi_4ds"
    max_length: int = 192
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.1
    seed: int = 42
    trust_remote_code: bool = False
    freeze_bottom_layers: int = 12
    mps_empty_cache_steps: int = 20
    eval_json_samples: int = 200
    save_predictions_jsonl: bool = False
    per_dataset_eval: bool = False
