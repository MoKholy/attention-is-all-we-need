import re


class WordLevelTokenizer:
    SOS = "[SOS]"
    EOS = "[EOS]"
    PAD = "[PAD]"

    def __init__(self, sentences: str = None, max_seq_len: int = 150) -> None:

        # token 2 index and index 2 token
        self.ttoi = {self.SOS: 0, self.EOS: 1, self.PAD: 2}
        self.itot = {i: t for t, i in self.ttoi.items()}
        self.vocab_len = 3
        self.max_seq_len = max_seq_len
        if sentences:
            for sentence in sentences:
                # word level tokenization
                for token in sentence:
                    if token not in self.ttoi:
                        self.itot[self.vocab_len], self.ttoi[token] = (
                            self.vocab_len,
                            token,
                        )
                        self.vocab_len += 1

    def tokenize(self, sentence: str, add_special_tokens: bool) -> list:

        # split into tokens of words, digits and punctuations
        tokens = re.findall(r"\w+|[^\w\s]+", sentence)

        if add_special_tokens:
            if len(tokens) > self.max_seq_len - 2:
                # end sentence at max_seq_len-2
                tokens = tokens[:-2]
            tokens = [self.SOS] + tokens + [self.EOS]
        else:
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:-2]
        return tokens

    def encode(
        self, sentences: list[str], pad: bool = True, add_special_tokens: bool = True
    ) -> list[int]:

        tokenized_sentences = [
            self.tokenize(sentence, add_special_tokens) for sentence in sentences
        ]
        encoded_sentences = [
            [self.ttoi[token] for token in tokenized_sentence]
            for tokenized_sentence in tokenized_sentences
        ]

        if pad:
            padded_sentences = []
            for sentence in encoded_sentences:
                n_pad_tokens = len(sentence) - self.max_seq_len
                padded_sentences.append([sentence] + n_pad_tokens * [self.PAD])
            encoded_sentences = padded_sentences
        return encoded_sentences
