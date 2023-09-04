import copy
import math

import torch
from torch import nn

from data_utils import FT_Dataset
from dataloader import DataLoader


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        assert u.shape == (x.shape[0], x.shape[1], 1)

        s = (x - u).pow(2).mean(-1, keepdim=True)
        assert s.shape == (x.shape[0], x.shape[1], 1)

        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.weight * x + self.bias


class Conv1D(nn.Module):
    """
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)

        x = x.view(-1, x.size(-1)) @ self.weight + self.bias
        assert x.shape == (x.shape[0], self.nf)

        x = x.view(*size_out)

        return x


class Attention(nn.Module):

    def __init__(self, config, scale=False):
        super(Attention, self).__init__()
        # nx = config.n_embd
        # n_state = config.n_embd
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(1, 1, config.n_ctx, config.n_ctx))
        self.n_head = config.n_head
        self.split_size = config.n_embd
        self.scale = scale
        self.c_attn = Conv1D(3 * config.n_embd, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        assert w.shape == (q.shape[0], q.shape[1], q.shape[2], k.shape[3])

        if self.scale:
            w = w / math.sqrt(v.size(-1))

        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]

        w_shape = w.shape
        w = w * b - 1e10 * (1 - b)
        assert w.shape == w_shape

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk = _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10)

        w = nn.Softmax(dim=-1)(w)
        result = torch.matmul(w, v)
        assert result.shape == (w.shape[0], w.shape[1], w.shape[2], v.shape[3])

        return result

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def forward(self, x, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        assert x.shape == (hidden_states.shape[0], hidden_states.shape[1], 3 * self.config.n_embd)

        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        len_kv = None

        if layer_past is not None:
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch, :, len_past, :] = key.squeeze(-1)
                past_value[_batch, :, len_past, :] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        assert present.shape == (2, value.shape[0], value.shape[1], value.shape[2], value.shape[3])

        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):

    def __init__(self, config, scale=False):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, scale)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * config.n_embd, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(nn.Module):

    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, past=None, len_past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if len_past is None:
            position_ids = torch.arange(
                past_length,
                input_ids.size(-1) + past_length,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1)

        assert position_ids.shape == input_ids.shape

        input_shape = input_ids.size()

        # Don't understand this part as it does not do any changes.
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        assert inputs_embeds.shape == (input_ids.shape[0], input_ids.shape[1], self.config.n_embd)

        position_embeds = self.wpe(position_ids)
        assert position_embeds.shape == (position_ids.shape[0], position_ids.shape[1], self.config.n_embd)

        hidden_states = inputs_embeds + position_embeds
        assert hidden_states.shape == (input_ids.shape[0], input_ids.shape[1], self.config.n_embd)

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            lora_attn_dim=0,
            lora_attn_alpha=128,
            lora_dropout=0.0,
            lora_r_dropout=0.0,
            fix_dropout=0.0,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        # self.initializer_range = initializer_range
        # self.lora_attn_dim = lora_attn_dim
        # self.lora_attn_alpha = lora_attn_alpha
        # self.lora_dropout = lora_dropout
        # self.lora_r_dropout = lora_r_dropout
        #
        # self.fix_dropout = fix_dropout


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2LMModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def forward(
        self,
        input_ids,
        lm_labels=None,
        lm_mask=None,
        past=None,
        len_past=None,
        label_smooth=0.0,
        is_report_accuracy=False
    ):
        _batch, _len = input_ids.shape
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:
            if label_smooth > 0.0001:
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

            loss = loss * lm_mask
            loss = loss.sum() / (lm_mask.sum() + 0.0001)

            return lm_logits, loss

        return lm_logits, presents


    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.startswith("transformer."):
                new_key = key[len("transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                assert False

        self.transformer.load_state_dict(state_dict, strict=False)
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)



if __name__ == '__main__':
    train_data = FT_Dataset("./data/e2e/train.jsonl", batch_size=8, max_seq_length=512)
    dataloader = DataLoader(dataset=train_data, batch_size=8)
    batch = next(iter(dataloader))
    config = GPT2Config(n_embd=768, n_layer=12, n_head=12)
    model = GPT2LMModel(config=config)
    # _, loss = model(
    #    input_ids=batch["input"],
    #    lm_labels=batch["target"],
    #    lm_mask=batch["mask"]
    # )
    # print(loss)

    # norm = LayerNorm(hidden_size=config.n_embd, eps=config.layer_norm_epsilon)
    # norm(x)
    #
    # conv = Conv1D(nf=config.n_embd, nx=config.n_embd)
    # conv(x)
    #
    # attn = Attention(config=config, scale=True)
    # attn(x)
    #
    # block = Block(config=config, scale=True)
    # block(x)

    cp = torch.load("./model/model.9000.pt", map_location=torch.device('cpu'))
    model.load_weight(cp)
