# src/text_tokenize.py
from __future__ import annotations
import re
from typing import List, Tuple

# simple, robust “word + punctuation” tokenizer with offsets from original text
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens, offsets = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets