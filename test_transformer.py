"""
Tests for ImplementTransformer.ipynb

These tests validate the core components of the Transformer implementation:
  - Config dataclass
  - MLP module
  - AttentionHead module
  - TransformerBlock module
  - TransformerArchitecture module
  - Text processing utilities (process_text, tokenize, encode, decode)
  - Data utilities (StackEncodedData, GetDataLoader)
  - Training loop (train)

Run with:
    pytest test_transformer.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Copy the source definitions here so tests are self-contained.
# If you refactor the notebook into a .py module, import from there instead.
# ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    d_model: int
    d_vocab: int
    d_hidden: int
    d_head: int
    t_blocks: int
    nc: int
    seed: int


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(config.d_model, config.d_hidden)
        self.fc2 = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.WQ = nn.Linear(config.d_model, config.d_head)
        self.WK = nn.Linear(config.d_model, config.d_head)
        self.WOV = nn.Linear(config.d_model, config.d_model)
        self.SM = nn.Softmax(dim=-1)

    def forward(self, x):
        nc = x.size(1)
        fullM = torch.full(size=(nc, nc), fill_value=float("-Inf"))
        maskM = torch.triu(fullM, diagonal=1)
        left_mult = self.WQ(x)
        right_mult = torch.transpose(self.WK(x), -2, -1)
        activation_input = left_mult @ right_mult + maskM
        activation_output = self.SM(activation_input)
        return activation_output @ self.WOV(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.AH1 = AttentionHead(config)
        self.MLP = MLP(config)

    def forward(self, x):
        attention_sum = x + self.AH1(x)
        return attention_sum + self.MLP(attention_sum)


class TransformerArchitecture(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.EC = nn.Embedding(num_embeddings=config.d_vocab, embedding_dim=config.d_model)
        self.PC = nn.Embedding(num_embeddings=config.nc, embedding_dim=config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.t_blocks)])
        self.UEMB = nn.Linear(config.d_model, config.d_vocab, bias=False)

    def forward(self, x):
        x = self.EC(x) + self.PC(torch.arange(x.size(1), device=x.device))
        for block in self.blocks:
            x = block(x)
        return self.UEMB(x)


def process_text(text, allowed_punctuation="-.,;:!?()\""+"".join(str(x) for x in range(10)),
                 punctuation_convert={'—': '-'}):
    import unicodedata
    for char, replacement in punctuation_convert.items():
        text = text.replace(char, replacement)
    text = '\n'.join(line for line in text.split('\n') if '.jpg' not in line)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.replace('\n', ' ').replace('\t', ' ')
    for char in allowed_punctuation:
        text = text.replace(char, f' {char} ')
    text = text.strip()
    while '  ' in text:
        text = text.replace('  ', ' ')
    text = ''.join(
        (char if (char.isalnum() or char in allowed_punctuation or char == ' ') else ' ')
        for char in text
    )
    text = text.lower()
    return text.strip()


def tokenize(text, process=False):
    if process:
        text = process_text(text)
    return text.split(' ')


def StackEncodedData(encodedData, config):
    n_tokens = (len(encodedData) // config.nc) * config.nc
    return encodedData[:n_tokens].reshape(-1, config.nc)


def GetDataLoader(data, batch_size, config):
    tensor_data = torch.tensor(np.array(data), dtype=torch.long)
    generator = torch.Generator(device=torch.device('cpu'))
    generator.manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=batch_size, generator=generator,
        shuffle=True, drop_last=bool(batch_size < len(data))
    )
    return loader


def train(model, optim, loader, num_epochs, config):
    losses = []
    model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            x_train = batch[:, :-1].long()
            y_train = batch[:, 1:].long()
            model_pred = model(x_train)
            loss = torch.nn.functional.cross_entropy(
                model_pred.transpose(1, 2), y_train, ignore_index=-1
            )
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
    return losses


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    """A tiny config that runs fast for testing."""
    return Config(d_model=16, d_vocab=50, d_hidden=32, d_head=8, t_blocks=1, nc=8, seed=42)


@pytest.fixture
def model(small_config):
    torch.manual_seed(small_config.seed)
    return TransformerArchitecture(small_config)


# ─────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_config_fields(self, small_config):
        assert small_config.d_model == 16
        assert small_config.d_vocab == 50
        assert small_config.d_hidden == 32
        assert small_config.d_head == 8
        assert small_config.t_blocks == 1
        assert small_config.nc == 8
        assert small_config.seed == 42

    def test_config_is_dataclass(self, small_config):
        assert isinstance(small_config, Config)


# ─────────────────────────────────────────────────────────────
# MLP tests
# ─────────────────────────────────────────────────────────────

class TestMLP:
    def test_output_shape(self, small_config):
        mlp = MLP(small_config)
        x = torch.randn(2, 5, small_config.d_model)
        out = mlp(x)
        assert out.shape == x.shape, "MLP should preserve input shape"

    def test_has_gelu(self, small_config):
        mlp = MLP(small_config)
        assert isinstance(mlp.gelu, nn.GELU)

    def test_linear_dimensions(self, small_config):
        mlp = MLP(small_config)
        assert mlp.fc1.in_features == small_config.d_model
        assert mlp.fc1.out_features == small_config.d_hidden
        assert mlp.fc2.in_features == small_config.d_hidden
        assert mlp.fc2.out_features == small_config.d_model


# ─────────────────────────────────────────────────────────────
# AttentionHead tests
# ─────────────────────────────────────────────────────────────

class TestAttentionHead:
    def test_output_shape(self, small_config):
        head = AttentionHead(small_config)
        batch, seq_len = 2, 5
        x = torch.randn(batch, seq_len, small_config.d_model)
        out = head(x)
        assert out.shape == (batch, seq_len, small_config.d_model)

    def test_causal_mask_applied(self, small_config):
        """Future tokens should not influence past tokens (causal masking)."""
        head = AttentionHead(small_config)
        head.eval()
        x = torch.randn(1, 4, small_config.d_model)
        # Replace last token with zeros and check first token is unchanged
        x_mod = x.clone()
        x_mod[0, -1] = 0.0
        with torch.no_grad():
            out1 = head(x)
            out2 = head(x_mod)
        # First token output should be identical because last token is masked
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), \
            "Causal mask should prevent future tokens from affecting earlier positions"

    def test_query_key_dimensions(self, small_config):
        head = AttentionHead(small_config)
        assert head.WQ.out_features == small_config.d_head
        assert head.WK.out_features == small_config.d_head


# ─────────────────────────────────────────────────────────────
# TransformerBlock tests
# ─────────────────────────────────────────────────────────────

class TestTransformerBlock:
    def test_output_shape(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(2, 5, small_config.d_model)
        out = block(x)
        assert out.shape == x.shape, "TransformerBlock should preserve input shape"

    def test_residual_connection(self, small_config):
        """Output should differ from a pure MLP or attention alone (residuals add)."""
        block = TransformerBlock(small_config)
        x = torch.randn(1, 4, small_config.d_model)
        out = block(x)
        assert not torch.allclose(out, x), "Block output should differ from identity"


# ─────────────────────────────────────────────────────────────
# TransformerArchitecture tests
# ─────────────────────────────────────────────────────────────

class TestTransformerArchitecture:
    def test_output_shape(self, small_config, model):
        batch, seq_len = 2, small_config.nc - 1
        x = torch.randint(0, small_config.d_vocab, (batch, seq_len))
        logits = model(x)
        assert logits.shape == (batch, seq_len, small_config.d_vocab)

    def test_num_blocks(self, small_config, model):
        assert len(model.blocks) == small_config.t_blocks

    def test_embedding_sizes(self, small_config, model):
        assert model.EC.num_embeddings == small_config.d_vocab
        assert model.EC.embedding_dim == small_config.d_model
        assert model.PC.num_embeddings == small_config.nc
        assert model.PC.embedding_dim == small_config.d_model

    def test_unembed_no_bias(self, small_config, model):
        assert model.UEMB.bias is None

    def test_forward_no_error(self, small_config, model):
        x = torch.randint(0, small_config.d_vocab, (1, 4))
        out = model(x)
        assert out is not None


# ─────────────────────────────────────────────────────────────
# Text processing tests
# ─────────────────────────────────────────────────────────────

class TestProcessText:
    def test_lowercases(self):
        assert process_text("Hello World") == "hello world"

    def test_removes_newlines(self):
        result = process_text("line one\nline two")
        assert '\n' not in result

    def test_removes_tabs(self):
        result = process_text("col1\tcol2")
        assert '\t' not in result

    def test_replaces_em_dash(self):
        result = process_text("well—known")
        assert '—' not in result
        assert '-' in result

    def test_spaces_around_punctuation(self):
        result = process_text("hello,world")
        assert 'hello , world' in result

    def test_removes_jpg_lines(self):
        text = "good line\nbad line with image.jpg\nanother good line"
        result = process_text(text)
        assert 'image' not in result
        assert 'good line' in result

    def test_no_double_spaces(self):
        result = process_text("too   many   spaces")
        assert '  ' not in result

    def test_returns_string(self):
        assert isinstance(process_text("test"), str)


class TestTokenize:
    def test_splits_on_space(self):
        tokens = tokenize("hello world foo")
        assert tokens == ["hello", "world", "foo"]

    def test_with_process(self):
        tokens = tokenize("Hello,World", process=True)
        assert "hello" in tokens
        assert "," in tokens

    def test_returns_list(self):
        assert isinstance(tokenize("a b c"), list)


# ─────────────────────────────────────────────────────────────
# Data utility tests
# ─────────────────────────────────────────────────────────────

class TestStackEncodedData:
    def test_output_shape(self, small_config):
        data = np.arange(100)
        result = StackEncodedData(data, small_config)
        assert result.shape[1] == small_config.nc
        assert result.shape[0] == 100 // small_config.nc

    def test_truncates_remainder(self, small_config):
        data = np.arange(101)  # 101 % 8 = 5 extra tokens
        result = StackEncodedData(data, small_config)
        total = result.shape[0] * result.shape[1]
        assert total <= 101


class TestGetDataLoader:
    def test_returns_dataloader(self, small_config):
        data = np.arange(64).reshape(8, 8)
        loader = GetDataLoader(data, 2, small_config)
        assert loader is not None

    def test_batch_shape(self, small_config):
        data = np.arange(64).reshape(8, 8)
        loader = GetDataLoader(data, 2, small_config)
        batch = next(iter(loader))
        assert batch.shape[0] == 2
        assert batch.shape[1] == 8


# ─────────────────────────────────────────────────────────────
# Training tests
# ─────────────────────────────────────────────────────────────

class TestTrain:
    def test_loss_decreases(self, small_config, model):
        """Loss should generally decrease over a short training run."""
        vocab_size = small_config.d_vocab
        nc = small_config.nc
        # Create synthetic data: shape (n_seqs, nc)
        n_seqs = 16
        data = np.random.randint(0, vocab_size, size=(n_seqs, nc))
        loader = GetDataLoader(data, 4, small_config)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = train(model, optim, loader, num_epochs=3, config=small_config)
        assert len(losses) > 0
        # First loss should be higher than last loss on average
        first_avg = np.mean(losses[:len(losses)//4])
        last_avg = np.mean(losses[3*len(losses)//4:])
        assert last_avg < first_avg, "Loss should decrease over training"

    def test_returns_loss_list(self, small_config, model):
        data = np.random.randint(0, small_config.d_vocab, size=(8, small_config.nc))
        loader = GetDataLoader(data, 2, small_config)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = train(model, optim, loader, num_epochs=1, config=small_config)
        assert isinstance(losses, list)
        assert all(isinstance(l, float) for l in losses)

    def test_loss_is_finite(self, small_config, model):
        data = np.random.randint(0, small_config.d_vocab, size=(8, small_config.nc))
        loader = GetDataLoader(data, 2, small_config)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = train(model, optim, loader, num_epochs=1, config=small_config)
        assert all(np.isfinite(l) for l in losses), "All losses should be finite"
