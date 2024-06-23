import torch


### Function to get causal masks
def get_causal_mask(seq):
    # B x seq_len
    _, seq_len = seq.size()
    ones = torch.ones(1, seq_len, seq_len, device=seq.device)
    causal_mask = torch.triu(ones, diagonal=1).bool()
    return causal_mask


def get_pad_mask(seq, pad_idx):

    return (seq == pad_idx).unsqueeze(-2)
