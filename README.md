# Transformer from Scratch — Amazon Food Reviews

## Contributors
This project was developed to complete the first project in Math 598C - Large Language Models at Colorado School of Mines. I worked along side Seth Dale, [@sddale] https://github.com/sddale/transformer , to independently, but in parallel, implement our own LLMs. We both discussed ideas for implementation and the theory behind the transformer architecture together. Implementation done separetly.

## Architecture Overview

The transformer is built from the following components:

| Class | Description |
|---|---|
| `Config` | Dataclass holding all hyperparameters |
| `MLP` | Two-layer feedforward network with GELU activation |
| `AttentionHead` | Single causal self-attention head with masking |
| `TransformerBlock` | One attention head + MLP with residual connections |
| `TransformerArchitecture` | Full model: token embedding + positional embedding + N blocks + unembedding |

---

## Setup & Reproduction

### 1. Clone the repository

### 2. Create the virtual environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml`, creates a `.venv` in the project folder, and installs all dependencies. To run any command inside the environment, prefix it with `uv run`, or activate the venv manually:

```bash
# Option A — prefix commands (recommended)
uv run jupyter notebook ImplementTransformer.ipynb
uv run pytest test_transformer.py -v

# Option B — activate manually
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Download the dataset

The notebook uses `kagglehub` to access the Amazon Fine Food Reviews dataset. Follow [Kaggle's setup guide](https://www.kaggle.com/docs/api) to place your `kaggle.json` credentials file at `~/.kaggle/kaggle.json`.

Then update the `data_path` variable in the notebook to point to where `kagglehub` downloads the data on your machine:

```python
data_path = "path/to/Reviews.csv"
```

Alternatively, download the CSV directly from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and set `data_path` accordingly.

### 4. Run the notebook

```bash
uv run jupyter notebook ImplementTransformer.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Load and preprocess the review text
2. Build a word-level vocabulary, excluding the 5000 longest/rarest words
3. Encode the data and re-encode a subset with a fresh filtered vocabulary
4. Define and train `REMY` (trained model) and instantiate `UNTRAINED_EMILLE` (baseline)
5. Generate and compare sample text from both models

---

## Key Hyperparameters

### Trained model (`REMY` — config)
`d_model` and `d_hidden` are derived from the filtered vocabulary: `DIM = int(sqrt(len(FILTERED_VOCAB)))`.

| Parameter | Value |
|---|---|
| `d_model` | `DIM` |
| `d_vocab` | size of filtered vocabulary |
| `d_hidden` | `DIM` |
| `d_head` | `int(DIM * 0.4)` |
| `t_blocks` | 2 |
| `nc` (context length) | 128 |
| `seed` | 2026 |

Training: 1 epoch, batch size 32, AdamW lr=1e-2, on ~1/60th of the data re-encoded with a fresh vocabulary (5000 longest words excluded).

### Untrained baseline (`UNTRAINED_EMILLE`)
An untrained `TransformerArchitecture` instance using the same `config`, used as a comparison baseline during text generation.

---

## Running the Tests

The test suite exercises all components **without** requiring the dataset. It uses small synthetic configs and data.

```bash
uv run pytest test_transformer.py -v
```

Expected output: all tests pass. The tests cover:
- `Config` field correctness
- `MLP` output shape and layer dimensions
- `AttentionHead` output shape and causal masking
- `TransformerBlock` shape preservation and residual connections
- `TransformerArchitecture` output shape, block count, embedding sizes
- `process_text` text normalization behavior
- `tokenize` splitting behavior
- `StackEncodedData` reshaping and truncation
- `GetDataLoader` batch shape
- `train` loss decrease, return types, and finiteness

---

## Expected Results

After training `REMY` for 1 epoch:
- Initial loss: ~11.1
- Final loss: ~5.4–5.7

Sample generation with prompt `"my favorite food is"`:
```
# REMY (trained)
my favorite food is . and and and and these and and ...

# UNTRAINED_EMILLE (baseline — random output)
my favorite food is four foamer foamer crosscontamination ...
```

---

## Requirements

- [uv](https://docs.astral.sh/uv/) (install via `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.12 (uv will fetch this automatically if not present)
- All Python dependencies are declared in `pyproject.toml` and installed via `uv sync`
