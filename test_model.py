import torch

from model import Attention, GPT2Config


def test_attention_split_heads():
    batch_size = 8
    seq_length = 512
    embedding_size = 768
    n_heads = 12
    x = torch.rand((batch_size, seq_length, embedding_size))
    config = GPT2Config(n_embd=embedding_size, n_layer=12, n_head=n_heads)
    attn = Attention(config=config, scale=True)

    assert attn.split_heads(x).shape == (batch_size, n_heads, seq_length, 64)

    assert attn.split_heads(x, k=True).shape == (batch_size, n_heads, 64, seq_length)
