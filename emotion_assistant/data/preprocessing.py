import re
from collections import Counter


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(text.split())

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab
