import torch
from datasets import load_dataset


### Function to get causal masks
def get_causal_mask(seq):
    # B x seq_len
    _, seq_len = seq.size()
    ones = torch.ones(1, seq_len, seq_len, device=seq.device)
    causal_mask = torch.triu(ones, diagonal=1).bool()
    return causal_mask


def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(-2)


### Data handling
def download_data(dataset_name="gsarti/iwslt2017_context", subset=None):
    save_dir = "./data/"
    if not subset:
        data = load_dataset(dataset_name, cache_dir=save_dir)
        load_dataset()
    else:
        data = load_dataset(dataset_name, subset, cache_dir=save_dir)
    return data
