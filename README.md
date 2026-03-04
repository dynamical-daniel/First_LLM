# Transformer from Scratch — Amazon Food Reviews

A from-scratch implementation of a decoder-only Transformer language model (inspired by GPT), trained on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle.

The model is nicknamed **REMY** (after the rat from Ratatouille 🐀).

---

## Project Structure

```
.
├── ImplementTransformer.ipynb   # Main notebook: model definition, training, generation
├── test_transformer.py          # Pytest unit tests for all core components
├── environment.yaml             # Conda environment specification
└── README.md                    # This file
```

---

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

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate the Conda environment

```bash
conda env create -f environment.yaml
conda activate transformer-env
```

### 3. Download the dataset

The notebook uses `kagglehub` to access the Amazon Fine Food Reviews dataset. You need a Kaggle account and API key configured. Follow [Kaggle's setup guide](https://www.kaggle.com/docs/api) to place your `kaggle.json` credentials file at `~/.kaggle/kaggle.json`.

Then update the `data_path` variable in the notebook to point to where `kagglehub` downloads the data on your machine:

```python
data_path = "path/to/Reviews.csv"
```

Alternatively, download the CSV directly from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and set `data_path` accordingly.

### 4. Run the notebook

```bash
jupyter notebook ImplementTransformer.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Load and preprocess the review text
2. Build a word-level vocabulary
3. Encode the data
4. Define and train two transformer models (`REMY` and `REMY2`) with different configs
5. Generate sample text from the trained models

---

## Key Hyperparameters

### Model 1 (`REMY` — config)
| Parameter | Value |
|---|---|
| `d_model` | 100 |
| `d_vocab` | vocab size − 1000 rare words |
| `d_hidden` | 200 |
| `d_head` | 100 |
| `t_blocks` | 2 |
| `nc` (context length) | 128 |
| `seed` | 2026 |

Training: 1 epoch, batch size 32, AdamW lr=1e-4, ~1/60th of the filtered data.

### Model 2 (`REMY2` — config2)
Uses a smaller subset (~1/10th of filtered data) re-encoded with a fresh vocabulary. `d_model` and `d_hidden` are set to `int(sqrt(vocab_size))`. Training: 1 epoch, batch size 32, AdamW lr=4e-3.

---

## Running the Tests

The test suite exercises all components **without** requiring the dataset. It uses small synthetic configs and data.

```bash
pytest test_transformer.py -v
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

After training for 1 epoch on the first model:
- Initial loss: ~12.2
- Final loss: ~6.3–6.5

After training for 1 epoch on the second model (`REMY2`):
- Initial loss: ~11.1
- Final loss: ~5.4–5.7

Sample generation from `REMY2` with prompt `"i do not like this because"`:
```
i do not like this because . and and and and these and and and these these snacks ...
```

---

## Requirements

- Python 3.12
- PyTorch 2.5.1
- NumPy, Pandas, Matplotlib
- jaxtyping
- kagglehub
- Jupyter
