import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):

    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, seq_len):
        super().__init__()
        self.data = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.seq_length = seq_len
        self.pad_token = torch.tensor()
