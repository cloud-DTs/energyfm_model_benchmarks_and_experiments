"""Chronicle tokenizer: a thin wrapper over a pickled tiktoken encoding."""

import os
import pickle
from functools import lru_cache


class ChronicleTokenizer:
    """131k-vocabulary BPE tokenizer (tiktoken encoding, trained from scratch)."""

    def __init__(self, enc, bos_token="<|bos|>"):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            return cls(pickle.load(f))

    def get_vocab_size(self):
        return self.enc.n_vocab

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(ids)
