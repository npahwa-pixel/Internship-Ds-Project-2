from __future__ import annotations
from typing import List
import torch
from nerjson.config.labels import ID2LABEL

def predict_word_tags(tokens: List[str], tokenizer, model, device: str, max_length: int) -> List[str]:
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    word_pred = [None] * len(tokens)
    for tok_i, wi in enumerate(word_ids):
        if wi is None: 
            continue
        if word_pred[wi] is None:
            word_pred[wi] = ID2LABEL[int(pred_ids[tok_i])]
    return [t if t is not None else "O" for t in word_pred]
