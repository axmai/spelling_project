# Georgian Spell Checker

A character-level neural spell checker for Georgian words, built with PyTorch. Given a misspelled word (a wrong letter, a missing or extra letter, or two swapped letters), the model predicts the correctly spelled word.

## Word list

The words used for training and evaluation are taken from [https://github.com/bumbeishvili/GeoWordsDatabase]([https://github.com/bumbeishvili/GeoWordsDatabase]).

Download the list and place it in the project root as `allwords-unique-sorted.txt` (or `allwords-unique-sorted.txt.zip`; the scripts can read either). The file is listed in `.gitignore`, so it is not stored in this repository.

## How it works

- **Training data:** the word list is filtered to Georgian-script words only (Unicode range U+10D0–U+10FF) with 2–20 letters, de-duplicated, and capped at the first 35,000 words. The file is assumed to be sorted by frequency, so the cap keeps the most common words.
- **Synthetic typos:** there is no hand-labelled error data. For every word the training script builds three pairs: one clean copy (word → word) and two corrupted copies (typo → word). Each corruption is one of:

  | Error type | Probability | Notes |
  |---|---|---|
  | Substitution | 40% | Replaced by a key next to it on the Georgian keyboard, or a random letter if none applies |
  | Transposition | 20% | Two neighbouring letters swapped |
  | Deletion | 25% | One letter removed |
  | Insertion | 15% | One random Georgian letter added |

- **Model:** a sequence-to-sequence network over characters.
  - Encoder: embedding layer (64 dimensions) followed by a single-layer GRU (128 hidden units).
  - Decoder: embedding layer plus a GRU that receives an attention-weighted context vector from the encoder at every step, then a linear layer over the vocabulary.
  - The corrected word is generated one character at a time until an end-of-word token is produced.
- **Training:** Adam (learning rate 1e-3) with cross-entropy loss, 90/10 train/validation split, batch size 256, up to 30 epochs, gradient clipping at 1.0, and a learning-rate scheduler that halves the rate when validation loss stalls. Teacher forcing starts at 0.7 and decays by a factor of 0.95 per epoch down to a minimum of 0.3. Training stops early after 10 epochs without improvement.

## Project structure

```
spelling_project/
├── train.py                          # builds the data, trains the model, saves checkpoints
├── train.ipynb                       # notebook version of training
├── inference.py                      # demo, evaluation and interactive spell checking
├── inference.ipynb                   # notebook version of inference
├── best_spellchecker.pt              # weights of the best epoch (lowest validation loss)
├── final_spellchecker_complete.pt    # full checkpoint: weights, vocabulary, sizes
└── .gitignore
```

Running `train.py` also writes `loss_curve.png` (training and validation loss per epoch).

## Requirements

- Python 3.9+
- `torch`
- `matplotlib`

```bash
pip install torch matplotlib
```

A GPU is used automatically for training if one is available; the model is small enough to train on CPU.

## Usage

### 1. Train

Make sure the word list is in the project root, then run:

```bash
python train.py
```

The script prints some sample training pairs, the vocabulary, and the loss for every epoch. When it finishes it saves:

- `best_spellchecker.pt`, the best weights seen during training
- `final_spellchecker_complete.pt`, the full checkpoint used by `inference.py`
- `loss_curve.png`, the loss curves

### 2. Run the spell checker

```bash
python inference.py
```

To use a different checkpoint or word list:

```bash
python inference.py path/to/checkpoint.pt path/to/allwords-unique-sorted.txt
```

The script runs in three stages:

1. **Demo:** corrects a fixed list of example words (clean words, substitutions, deletions, transpositions and insertions) and prints input, output, expected word and a ✓/✗ for each.
2. **Evaluation (optional, asks `[y/N]`):** corrupts 500 words from the word list with a mix of clean words and the four error types, and reports overall accuracy, accuracy per error type, and a few sample failures.
3. **Interactive mode (optional, asks `[y/N]`):** type a Georgian word and get the corrected version back. Enter `quit`, `exit` or `q` to leave.

### Using it from Python

```python
from inference import correct_word

print(correct_word("სკოა", "final_spellchecker_complete.pt"))  # "სკოა" is a typo of "სკოლა"
```

The checkpoint is loaded once and cached for later calls.

## Configuration

The main settings are constants at the top of `train.py`:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_WORDS` | `35_000` | Number of words used (`None` for all) |
| `MIN_WORD_LEN` / `MAX_WORD_LEN` | `2` / `20` | Word length filter |
| `ERRORS_PER_WORD` | `2` | Corrupted copies generated per word |
| `VAL_RATIO` | `0.10` | Fraction of pairs held out for validation |
| `BATCH_SIZE` | `256` | Batch size |
| `EMBED_DIM` / `HIDDEN_DIM` | `64` / `128` | Embedding and GRU sizes |
| `N_EPOCHS` | `30` | Maximum number of epochs |
| `LR` | `1e-3` | Initial learning rate |
| `TEACHER_FORCING`, `TF_DECAY`, `TF_MIN` | `0.7`, `0.95`, `0.3` | Teacher-forcing schedule |
| `PATIENCE` | `10` | Epochs without improvement before early stopping |
| `SEED` | `42` | Random seed |

## Notes

- `inference.py` redefines the model classes so it can run on its own. If you change the architecture in `train.py`, make the same change in `inference.py`, or the saved weights will not load.
- Only single-word corrections are supported. The model was trained on words with at most one typo each, so words with several errors may not be corrected reliably.
- The vocabulary comes from the training words, so characters outside the Georgian alphabet are treated as unknown.
