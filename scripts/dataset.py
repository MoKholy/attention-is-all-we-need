import torch
from torch.utils.data import Dataset

from utils import get_causal_mask, get_pad_mask


class GPT_Tokenizer_Dataset(Dataset):

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_len=300,
        split="train",
        src_lang="en",
        tgt_lang="ar",
    ):
        """dataset is a huggingface Dataset. tokenizer should be passed along, seq_len will be maximum seq_len in dataset"""

        super().__init__()
        self.data = dataset[split]
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len
        self.bos_token = torch.tensor(
            tokenizer.encode("<|BOS|>", allowed_special="all")
        )
        self.eos_token = torch.tensor(
            tokenizer.encode("<|EOS|>", allowed_special="all")
        )
        self.pad_token = torch.tensor(
            tokenizer.encode("<|PAD|>", allowed_special="all")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_tgt_pair = self.data[idx]["translation"]
        src_text = src_tgt_pair[self.src_lang]
        tgt_text = src_tgt_pair[self.tgt_lang]
        src_tokens = self.tokenizer.encode(src_text)
        tgt_tokens = self.tokenizer.encode(tgt_text)

        enc_n_pad_tokens = self.max_seq_len - len(src_tokens) - 2  # add bos and eos
        dec_n_pad_tokens = (
            self.max_seq_len - len(tgt_tokens) - 1
        )  # add bos, eos is in label

        enc_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_n_pad_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        dec_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(tgt_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_n_pad_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # add label
        label = torch.cat(
            [
                torch.tensor(tgt_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_n_pad_tokens, dtype=torch.int64),
            ]
        )
        assert enc_input.size(0) == self.max_seq_len
        assert dec_input.size(0) == self.max_seq_len
        assert label.size(0) == self.max_seq_len

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "encoder_mask": get_pad_mask(
                enc_input.unsqueeze(0), self.pad_token[0].item()
            ).shape,
            "decoder_mask": (
                get_causal_mask(dec_input)
                | get_pad_mask(dec_input.unsqueeze(0), self.pad_token[0].item())
            ).shape,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
