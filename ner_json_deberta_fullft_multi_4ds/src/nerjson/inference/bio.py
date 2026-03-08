from __future__ import annotations
from typing import List, Tuple

def reconstruct_text_and_token_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    offsets, parts, cur = [], [], 0
    for idx, tok in enumerate(tokens):
        if idx > 0:
            parts.append(" "); cur += 1
        s = cur
        parts.append(tok); cur += len(tok)
        offsets.append((s, cur))
    return "".join(parts), offsets

def spans_from_bio(tags: List[str], token_offsets: List[Tuple[int, int]]):
    spans = []
    cur_type = cur_start = cur_end = None
    def flush():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None:
            spans.append((cur_type, cur_start, cur_end))
        cur_type = cur_start = cur_end = None
    for tag, (s, e) in zip(tags, token_offsets):
        if tag == "O":
            flush(); continue
        pref, typ = tag.split("-", 1)
        if pref == "B" or cur_type != typ:
            flush(); cur_type, cur_start, cur_end = typ, s, e
        else:
            cur_end = e
    flush()
    return spans
