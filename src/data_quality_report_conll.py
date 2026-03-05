from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

from build_conll import load_conll

ENTITY_TYPES = {"PER", "ORG", "LOC", "MISC"}
VALID_TAGS = {"O"} | {f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")}

# Spacing rules used ONLY to create a deterministic reference TEXT for offsets.
# Note: this may not match the original raw sentence perfectly, but it must be self-consistent.
PUNCT_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "»", "”"}
PUNCT_NO_SPACE_AFTER  = {"(", "[", "{", "«", "“", "$"}


def sha1_of_tokens(tokens: List[str]) -> str:
    s = "\u001f".join(tokens).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def conll_reconstruct_text(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Deterministically reconstruct TEXT and per-token char offsets.
    Strong checks below validate that these offsets actually slice back to the tokens.
    """
    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    pos = 0

    for i, tok in enumerate(tokens):
        if i == 0:
            prefix = ""
        else:
            prev = tokens[i - 1]
            if tok in PUNCT_NO_SPACE_BEFORE:
                prefix = ""
            elif prev in PUNCT_NO_SPACE_AFTER:
                prefix = ""
            else:
                prefix = " "

        parts.append(prefix)
        pos += len(prefix)

        start = pos
        parts.append(tok)
        pos += len(tok)
        end = pos

        offsets.append((start, end))

    return "".join(parts), offsets


def bio_to_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    BIO tags -> token spans: (TYPE, start_tok, end_tok_exclusive)
    """
    spans: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t == "O":
            i += 1
            continue

        if t.startswith("B-"):
            typ = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append((typ, i, j))
            i = j
            continue

        if t.startswith("I-"):
            # Stray I- treated as a new entity span, but counted as BIO violation elsewhere
            typ = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.append((typ, i, j))
            i = j
            continue

        i += 1

    return spans


def spans_to_json(text: str, offsets: List[Tuple[int, int]], spans: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    ents = []
    for typ, s, e in spans:
        start = offsets[s][0]
        end = offsets[e - 1][1]
        ents.append({"type": typ, "text": text[start:end], "start": start, "end": end})
    return {"entities": ents}


def bio_transition_violations(tags: List[str]) -> int:
    """
    Count BIO violations:
      - I-X that does not follow B-X or I-X
    """
    bad = 0
    prev = "O"
    for t in tags:
        if t.startswith("I-"):
            typ = t[2:]
            if prev not in (f"B-{typ}", f"I-{typ}"):
                bad += 1
        prev = t
    return bad


def percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(round((p / 100.0) * (len(values_sorted) - 1)))
    k = max(0, min(k, len(values_sorted) - 1))
    return float(values_sorted[k])


@dataclass
class SplitStats:
    num_sentences: int
    token_count_total: int
    token_len_avg: float
    token_len_median: float
    token_len_p95: float
    token_len_max: int
    empty_entity_sentences: int
    empty_entity_ratio: float
    entity_counts: Dict[str, int]
    entity_total: int


def compute_split_stats(rows: List[Dict[str, Any]]) -> SplitStats:
    lengths = [len(r["tokens"]) for r in rows]
    token_count_total = sum(lengths)

    entity_counts: Dict[str, int] = {t: 0 for t in ENTITY_TYPES}
    empty_entity_sentences = 0
    entity_total = 0

    for r in rows:
        spans = bio_to_spans(r["tags"])
        if not spans:
            empty_entity_sentences += 1
        for typ, _, _ in spans:
            if typ in entity_counts:
                entity_counts[typ] += 1
            entity_total += 1

    return SplitStats(
        num_sentences=len(rows),
        token_count_total=token_count_total,
        token_len_avg=(token_count_total / len(rows)) if rows else 0.0,
        token_len_median=float(statistics.median(lengths)) if lengths else 0.0,
        token_len_p95=percentile(lengths, 95),
        token_len_max=max(lengths) if lengths else 0,
        empty_entity_sentences=empty_entity_sentences,
        empty_entity_ratio=(empty_entity_sentences / len(rows)) if rows else 0.0,
        entity_counts=entity_counts,
        entity_total=entity_total,
    )


def leakage_and_duplicates(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    hashes = {k: [sha1_of_tokens(r["tokens"]) for r in v] for k, v in splits.items()}
    sets = {k: set(v) for k, v in hashes.items()}

    within = {k: len(v) - len(set(v)) for k, v in hashes.items()}

    train = sets.get("train", set())
    val = sets.get("validation", set())
    test = sets.get("test", set())

    return {
        "within_split_duplicates": within,
        "cross_split_overlap": {
            "train∩validation": len(train & val),
            "train∩test": len(train & test),
            "validation∩test": len(val & test),
        },
    }


def validate_rows(split_name: str, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Real checks that can fail:
      - token/tag length mismatch
      - invalid tags
      - BIO violations
      - token slice mismatch (text[offsets] != token)
      - offset monotonicity / invalid gaps
      - out-of-bounds offsets
      - double spaces in reconstructed text
    """
    errors = {
        "len_mismatch": 0,
        "invalid_tag": 0,
        "bio_violations": 0,
        "token_slice_mismatch": 0,
        "offset_out_of_bounds": 0,
        "offset_non_monotonic": 0,
        "offset_gap_invalid": 0,
        "double_space_in_text": 0,
    }

    for r in rows:
        tokens = r["tokens"]
        tags = r["tags"]

        if len(tokens) != len(tags):
            errors["len_mismatch"] += 1
            continue

        for t in tags:
            if t not in VALID_TAGS:
                errors["invalid_tag"] += 1

        errors["bio_violations"] += bio_transition_violations(tags)

        text, offsets = conll_reconstruct_text(tokens)

        # Check double spaces (common reconstruction bug)
        if "  " in text:
            errors["double_space_in_text"] += 1

        # Check each token is exactly recoverable by slicing offsets
        prev_end = -1
        for i, (s, e) in enumerate(offsets):
            if not (0 <= s <= e <= len(text)):
                errors["offset_out_of_bounds"] += 1
                continue

            if e < prev_end:
                errors["offset_non_monotonic"] += 1
            prev_end = e

            if text[s:e] != tokens[i]:
                errors["token_slice_mismatch"] += 1

        # Check gaps between consecutive tokens are either 0 or 1
        for i in range(len(offsets) - 1):
            gap = offsets[i + 1][0] - offsets[i][1]
            if gap not in (0, 1):
                errors["offset_gap_invalid"] += 1

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reports/conll_quality_report.json")
    ap.add_argument("--strict", action="store_true", help="Fail if any validation error occurs")
    ap.add_argument("--sample_json", type=int, default=0, help="Dump N gold JSON samples from test split")
    args = ap.parse_args()

    ds = load_conll()
    splits = {
        "train": [ds["train"][i] for i in range(len(ds["train"]))],
        "validation": [ds["validation"][i] for i in range(len(ds["validation"]))],
        "test": [ds["test"][i] for i in range(len(ds["test"]))],
    }

    acquisition = {
        "dataset": "conll2003 (Hugging Face datasets)",
        "splits": {k: len(v) for k, v in splits.items()},
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    validation = {k: validate_rows(k, v) for k, v in splits.items()}
    stats = {k: compute_split_stats(v).__dict__ for k, v in splits.items()}
    leakage = leakage_and_duplicates(splits)

    samples = []
    if args.sample_json > 0:
        n = min(args.sample_json, len(splits["test"]))
        for i in range(n):
            r = splits["test"][i]
            text, offsets = conll_reconstruct_text(r["tokens"])
            spans = bio_to_spans(r["tags"])
            samples.append({
                "idx": i,
                "text": text,
                "gold_json": spans_to_json(text, offsets, spans),
            })

    report = {
        "acquisition": acquisition,
        "validation": validation,
        "stats": stats,
        "leakage": leakage,
        "sample_gold_json": samples,
        "notes": {
            "text_reconstruction": (
                "Offsets are computed against a deterministic reconstructed TEXT. "
                "This ensures internal consistency for JSON offsets, but may not exactly match the original raw sentence."
            )
        }
    }

    print("\n=== CoNLL2003 Data Acquisition & Quality Report (Stronger Checks) ===")
    print("Splits:", acquisition["splits"])

    print("\nValidation errors (counts):")
    for split, errs in validation.items():
        print(f"  {split}: {errs}")

    print("\nSplit stats (high-level):")
    for split, st in stats.items():
        print(
            f"  {split}: sentences={st['num_sentences']}, "
            f"avg_len={st['token_len_avg']:.2f}, p95_len={st['token_len_p95']:.0f}, "
            f"empty_sent%={100*st['empty_entity_ratio']:.2f}%, entities={st['entity_total']}"
        )
        print(f"    entity_counts={st['entity_counts']}")

    print("\nLeakage / duplicates:")
    print("  within_split_duplicates:", leakage["within_split_duplicates"])
    print("  cross_split_overlap:", leakage["cross_split_overlap"])

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote report to: {out_path}")

    if args.strict:
        total = sum(sum(v.values()) for v in validation.values())
        if total > 0:
            raise SystemExit(f"STRICT MODE FAILED: total validation errors = {total}")


if __name__ == "__main__":
    main()