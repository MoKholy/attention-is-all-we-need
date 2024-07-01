"""Simple BPE tokenizer. Follows Andrej Karpathy's minBPE:
https://github.com/karpathy/minbpe
"""

import unicodedata


def get_freqs(token_ids, counts=None):
    """Given a list of token ids, return a dictionary of the frequencies of consecutive pairs. Optionally, allow updating an exisiting dictionary 'counts'"""
    counts = {} if counts is None else counts

    for pair in zip(token_ids, token_ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_tokens(token_ids, pair, idx):
    """Given an original list of token ids, merge a given pair of tokens by substituting them with passed idx.
    Example:
    token_ids: [1, 1, 2, 1, 1], pair: (1, 1), idx: 3 -> token_ids: [3, 2, 3]
    """
    new_ids = []
    i = 0
    p0, p1 = pair
    while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == p0 and token_ids[i + 1] == p1:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(token_ids[i])
            i += 1
    return new_ids


def render_token(s: bytes):
    # decode to string first, and remove control chars while printing
    s = s.decode("utf-8", errors="replace")
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


class BaseTokenizer:
    """Base class for bpe tokenizer"""

    def __init__(self):
        self.merges = {}
        self.pattern = ""  # optional regex pattern
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        """Tokenizer can train a vocabulary of size vocab_size from text"""
        raise NotImplementedError

    def encode(self, text):
        """Given text, encode the text as a list of integers based on learnt vocabulary"""
        raise NotImplementedError

    def decode(self, ids):
        """Given integer ids, transforms them into a string"""
        raise NotImplementedError

    def _build_vocab(self):
        """Build vocab based on merges"""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """Saves tokenizer into two files, model and vocab file. Vocab file is meant for human inspection, while model file is intended for load().
        Files saved:
        - file_prefix.model
        - file_prefix.vocab
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w") as model_f:
            # simple header
            model_f.write("Simple BPE Tokenizer \n")
            # regex pattern
            model_f.write(f"{self.pattern}\n")
            # len of special tokens
            model_f.write(f"{len(self.special_tokens)}\n")
            # special tokens and their indices
            for special_token, idx in self.special_tokens.items():
                model_f.write(f"{special_token} {idx}\n")

            # merges dictionary
            for idx0, idx1 in self.merges:
                model_f.write(f"{idx0} {idx1}\n")

        vocab_file = file_prefix + ".vocab"
        inverse_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w") as vocab_f:
            for idx, token in self.vocab.items():

                # handle partial utf-8 characters. Not all byte sequences are valid utf-8 characters.
                s = render_token(token)

                if idx in inverse_merges:
                    # if token has children, render them as a merge
                    idx0, idx1 = inverse_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    vocab_f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    vocab_f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """loads model from model_file save file."""
        assert model_file.endswith(".model")

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as model_f:
            # read header
            header = model_f.readline().strip()
            # read pattern
            self.pattern = model_f.readline().strip()

            # read num of special tokens and then read special tokens
            n_special_tokens = int(model_f.readline().strip())

            for i in range(n_special_tokens):
                special_token, special_token_idx = model_f.readline().strip().split()
                special_tokens[special_token] = special_token_idx

            # read merges
            for line in model_f:
                idx0, idx1 = map(int, line.split())
                merges[(idx0, idx1)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab


class BasicBPETokenizer(BaseTokenizer):
    """Basic BPE tokenizer."""

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        n_merges = vocab_size - 256

        # changes text to bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(n_merges):

            byte_freqs = get_freqs(ids)
            # find pair with max freq
            pair = max(byte_freqs, key=byte_freqs.get)

            idx = 256 + i
            ids = merge_tokens(ids, pair, idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge #{i+1}/{n_merges}: {pair} ---> {idx} ({vocab[idx]}) had {byte_freqs[pair]} occurrences."
                )
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text): ...
