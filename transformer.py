import unittest

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embed_dim: int,
        input_seq_len: int,
        output_seq_len: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        attn_dropout: float,
        ffn_dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_vocab_size,
            embed_dim,
            input_seq_len,
            n_heads,
            ffn_dim,
            n_layers,
            attn_dropout,
            ffn_dropout,
            eps,
        )
        self.decoder = Decoder(
            output_vocab_size,
            embed_dim,
            output_seq_len,
            n_heads,
            ffn_dim,
            n_layers,
            attn_dropout,
            ffn_dropout,
            eps,
        )
        self.project = nn.Linear(embed_dim, output_vocab_size)

        # init weights using xavier_uniform
        self._reset_parameters()

    def encode(self, x, input_mask):
        x = self.encoder(x, input_mask)
        return x

    def decode(self, encoder_out, input_mask, x, output_mask):
        if not encoder_out:
            assert (
                input_mask is not None
            ), "Neither encoder_out, nor input_mask are passed"
            encoder_out = self.encoder(x, input_mask)
        x = self.decoder(x, encoder_out, input_mask, output_mask)
        return x

    def get_logits(self, x):
        x = self.project(x)
        return x

    # init parameters using xavier uniform
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TestTransformer(unittest.TestCase):

    def test_transformer_init(self):
        # load config file
        import yaml

        with open("configs/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        general = config["model"]["general"]
        encoder = config["model"]["encoder"]
        decoder = config["model"]["decoder"]
        transformer = Transformer(**general, **encoder, **decoder)
        # print(f"{transformer.modules = }")

    # TODO
    def test_transformer_inf(self):
        # load config file
        import yaml

        with open("configs/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        general = config["model"]["general"]
        encoder = config["model"]["encoder"]
        decoder = config["model"]["decoder"]
        with torch.no_grad():
            transformer = Transformer(**general, **encoder, **decoder)
            transformer.eval()

            # create input tensor
            # let 9 be pad token idx
            encoder_input = torch.IntTensor([1, 2, 3, 4, 9, 9, 9])
            input_mask = encoder_input != 9
            encoder_output = transformer.encoder(encoder_input, input_mask)
            # check encoder output has no NaNs
            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)
            # check output dims are correct

            # B=1, seq_len = 7, embed_dim = 16
            self.assertEqual(torch.Size([1, 7, 16]), encoder_output.shape)

            # create output_mask
            # create output mask
            # both equivalent
            output_mask = torch.tril(torch.ones((1, 7, 7))) == 1
            mask = torch.triu(torch.ones((1, 7, 7)), diagonal=1).type(torch.int) == 0
            # create decoder input, assume start of speech token is 1
            decoder_input = torch.IntTensor([[1, 2, 3, 4, 7, 7, 6]])
            for i in range(7):  # max decode length
                decoder_output = transformer.decoder(
                    decoder_input, encoder_output, input_mask, output_mask
                )

                print(f"{decoder_output.shape = }")
                break


if __name__ == "__main__":

    # test that it works
    unittest.main()
