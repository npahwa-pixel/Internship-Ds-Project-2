from __future__ import annotations

from typing import Dict, List, Tuple

from jsonschema import validate as jsonschema_validate, ValidationError

from nerjson.config.labels import TYPE_MAP
from nerjson.config.schema import NER_JSON_SCHEMA
from nerjson.inference.bio import reconstruct_text_and_token_offsets, spans_from_bio


def tokens_to_json(tokens: List[str], pred_tags: List[str]) -> Dict:
    text, offsets = reconstruct_text_and_token_offsets(tokens)
    spans = spans_from_bio(pred_tags, offsets)
    entities = []
    for typ, s, e in spans:
        entities.append({"type": TYPE_MAP.get(typ, "MISC"), "text": text[s:e], "start": int(s), "end": int(e)})
    return {"entities": entities}


def json_validity_rate(samples: List[Tuple[List[str], List[str]]]) -> float:
    ok = 0
    for tokens, pred_tags in samples:
        obj = tokens_to_json(tokens, pred_tags)
        try:
            jsonschema_validate(instance=obj, schema=NER_JSON_SCHEMA)
            ok += 1
        except ValidationError:
            pass
    return ok / max(1, len(samples))
