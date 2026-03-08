from __future__ import annotations
import numpy as np
from seqeval.metrics import f1_score as seqeval_f1
from nerjson.config.labels import ID2LABEL

def compute_seqeval_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for p, l in zip(preds, labels):
        seq_l, seq_p = [], []
        for pi, li in zip(p, l):
            if li == -100:
                continue
            seq_l.append(ID2LABEL[int(li)])
            seq_p.append(ID2LABEL[int(pi)])
        true_labels.append(seq_l); true_preds.append(seq_p)
    return {"f1": float(seqeval_f1(true_labels, true_preds))}
