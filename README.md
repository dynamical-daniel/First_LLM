# Transformer from Scratch — Amazon Food Reviews

A vanilla transformer architecture LLM trained on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle.

The trained model is nicknamed **REMY** (after the rat from Ratatouille 🐀), and is compared against **UNTRAINED_EMILLE**.

## Collaborators
This project was completed to fulfill project 1 of Math 598C - Large Language Models @ Colorado School of Mines. I collaborated with [@sddale] throughout the entirety of the project. We both shared ideas for implementation and discussed the theory of the architecture of our models. We discussed issues we faced preprocessing the data and creating our own training set. Implementation of the models was accomplished independently.

## Architecture Overview

| Class | Description |
|---|---|
| `Config` | Configurable set of model hyperparameters |
| `MLP` | Standard Multi-Layer Perceptron used to predict next word |
| `AttentionHead` | Self-attention head with upper-triangular masking (no peeking!) |
| `TransformerBlock` | AttentionHead + MLP |
| `TransformerArchitecture` | Token embedding + positional embedding + N blocks + unembedding |

---

## Setup & Reproduction

### 1. Clone the repository
Everyone can do this.

### 2. Install uv and sync dependencies
Install all project dependencies:

```bash
uv sync
```

### 3. Download the dataset

The notebook uses `kagglehub` to download the Amazon Fine Food Reviews dataset. Follow [Kaggle's API setup guide](https://www.kaggle.com/docs/api). OR,

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
# Replace with your actual path
data_path = "/home/<your-username>/.cache/kagglehub/datasets/snap/amazon-fine-food-reviews/versions/2/Reviews.csv"
```

### 5. Run the notebook

```bash
uv run jupyter notebook ImplementTransformer.ipynb
```

Run all cells in the notebook. The notebook will:
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
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/61e229c5-8681-4de6-b219-59e18c88a4d7" />

Sample generation with prompt `"my favorite food is"` and `max_new_tokens=50`:

```
# REMY (trained)
my favorite food is samples artificial samples samples food know samples food know samples samples samples food know samples samples food fun whole premium i artificial samples labrador samples without know samples samples samples food know samples fun samples four food know samples crispy food know samples samples samples fun samples four food samples

# UNTRAINED_EMILLE
my favorite food is four four four four four four four four four four four four four four four four four four four confusion four four four four four four confusion four four four four four four four four four confusion four four confusion four four four confusion four four four four four four
```

---

## Conclusion
My trained LLM did not produce the greatest response. I believe this is from prioritizing manageable training times (no more than 20 minutes, with trainings usually taking from 10 - 15 minutes). More work could have been done to preprocess the data. The data is messy with typos and unique tokens that quickly expand the size of the vocabulary. 

There are many strucural changes that can be made to the LLM to likely improve performance more so than just hyperparameter tuning.
