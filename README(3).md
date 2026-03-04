# Transformer from Scratch — Amazon Food Reviews

A from-scratch implementation of a decoder-only Transformer language model (inspired by GPT), trained on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle.

The trained model is nicknamed **REMY** (after the rat from Ratatouille 🐀), and is compared against **UNTRAINED_EMILLE**, an uninitialized model with the same architecture used as a random baseline.

---

## Project Structure

```
.
├── ImplementTransformer.ipynb      # Main notebook: model, training, generation
├── test_transformer.py             # Pytest unit tests for all core components
├── pyproject.toml                  # uv project and dependency specification
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: runs tests on every push
└── README.md                       # This file
```

---

## Architecture Overview

| Class | Description |
|---|---|
| `Config` | Dataclass holding all hyperparameters |
| `MLP` | Two-layer feedforward network with GELU activation |
| `AttentionHead` | Single causal self-attention head with upper-triangular masking |
| `TransformerBlock` | AttentionHead + MLP with residual connections |
| `TransformerArchitecture` | Token embedding + positional embedding + N blocks + unembedding |

---

## Setup & Reproduction

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install uv and sync dependencies

If you don't have [uv](https://docs.astral.sh/uv/) installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via pip
pip install uv
```

Then install all project dependencies:

```bash
uv sync
```

This creates a `.venv` in the project folder and installs everything declared in `pyproject.toml`.

### 3. Download the dataset

The notebook uses `kagglehub` to download the Amazon Fine Food Reviews dataset. You will need a Kaggle account with an API key configured. Follow [Kaggle's API setup guide](https://www.kaggle.com/docs/api) to place your `kaggle.json` credentials at `~/.kaggle/kaggle.json`.

Cell 3 of the notebook runs the download:

```python
kagglehub.dataset_download("snap/amazon-fine-food-reviews")
```

`kagglehub` will print the path where the dataset was saved. It is typically something like:

```
# macOS / Linux
~/.cache/kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/

# Windows
C:\Users\<your-username>\.cache\kagglehub\datasets\snap\amazon-fine-food-reviews\versions\2\
```

### 4. Update `data_path`

After running the download cell, **edit the `data_path` variable in Cell 4** to match the path printed by `kagglehub` on your machine, pointing to `Reviews.csv`:

```python
# Example — replace with your actual path
data_path = "/home/<your-username>/.cache/kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/Reviews.csv"
```

The current value in the notebook is the original author's local Windows path and **will not work on your machine without this change.**

### 5. Run the notebook

```bash
uv run jupyter notebook ImplementTransformer.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Download the dataset (Cell 3) and load `Reviews.csv` from `data_path` (Cell 4)
2. Clean and preprocess the review text
3. Build a word-level vocabulary, excluding the 1000 longest/rarest words
4. Re-encode a 1/60th subset of the data with a fresh filtered vocabulary
5. Define `config`, then instantiate and train `REMY` for 1 epoch
6. Instantiate `UNTRAINED_EMILLE` with the same `config` as a random baseline
7. Plot the training loss curve
8. Generate text from both models and compare outputs

---

## Key Hyperparameters (`config`)

`DIM` is computed as `int(sqrt(len(FILTERED_VOCAB)))` after vocabulary filtering.

| Parameter | Value |
|---|---|
| `d_model` | `DIM` |
| `d_vocab` | size of filtered vocabulary |
| `d_hidden` | `DIM` |
| `d_head` | `int(DIM * 0.4)` |
| `t_blocks` | 2 |
| `nc` (context length) | 128 |
| `max_tokens` | 50 |
| `seed` | 2026 |

Training: 1 epoch, batch size 32, AdamW lr=1e-2.

---

## Expected Results

After training `REMY` for 1 epoch:
- Initial loss: ~11.1
- Final loss: ~5.4–5.7

Sample generation with prompt `"my favorite food is"` and `max_new_tokens=50`:

```
# REMY (trained)
my favorite food is the best . i have been using this for a long time and it is ...

# UNTRAINED_EMILLE (random baseline)
my favorite food is four foamer foamer crosscontamination schmoo foamer four schmoo ...
```

---

## Running the Tests

The test suite covers all components and runs **without requiring the dataset**.

```bash
uv run pytest test_transformer.py -v
```

Tests cover: `Config` fields, `MLP` shapes, `AttentionHead` causal masking, `TransformerBlock` residuals, `TransformerArchitecture` output shape, `process_text` / `tokenize` behavior, `StackEncodedData`, `GetDataLoader`, and the `train` loop.

---

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions (see `.github/workflows/ci.yml`). Results are visible under the **Actions** tab of the repository.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Python 3.12 (fetched automatically by uv if not present)
- A [Kaggle account](https://www.kaggle.com) with API credentials for dataset download
- All Python dependencies are declared in `pyproject.toml` and installed via `uv sync`
