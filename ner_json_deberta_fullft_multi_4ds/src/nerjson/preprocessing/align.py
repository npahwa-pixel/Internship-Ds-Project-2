from __future__ import annotations

def tokenize_and_align_labels(tokenizer, examples, max_length: int):
    enc = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = enc.word_ids(batch_index=i)
        word_labels = examples["tags_unified"][i]
        aligned = []
        prev = None
        for w in word_ids:
            if w is None:
                aligned.append(-100)
            elif w != prev:
                aligned.append(int(word_labels[w]))
            else:
                aligned.append(-100)
            prev = w
        labels.append(aligned)
    enc["labels"] = labels
    return enc
