import os
import re
import random
import pickle
import zipfile
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

RAW_WORDS_PATH = "allwords-unique-sorted.txt"   # or .zip
RAW_WORDS_ZIP = "allwords-unique-sorted.txt.zip"
OUTPUT_MODEL = "final_spellchecker_complete.pt"
OUTPUT_PLOT = "loss_curve.png"

MIN_WORD_LEN = 2
MAX_WORD_LEN = 20
MAX_WORDS = 35_000 # cap to keep training tractable; set None for all
ERRORS_PER_WORD = 2 # corrupted copies per word (+ 1 clean copy)
VAL_RATIO = 0.10

BATCH_SIZE = 256
EMBED_DIM = 64
HIDDEN_DIM = 128
N_EPOCHS = 30
LR = 1e-3
TEACHER_FORCING = 0.7
TF_DECAY = 0.95
TF_MIN = 0.3
CLIP = 1.0
PATIENCE = 10 # early stopping


GEORGIAN_RE = re.compile(r'^[\u10D0-\u10FF]+$')   # only Georgian script letters

def load_words(path_txt, path_zip, max_words=None):
    raw_bytes = None

    if os.path.exists(path_txt):
        print(f"Reading {path_txt} …")
        with open(path_txt, "rb") as f:
            raw_bytes = f.read()
    elif os.path.exists(path_zip):
        print(f"Extracting {path_zip} …")
        with zipfile.ZipFile(path_zip) as z:
            name = [n for n in z.namelist() if n.endswith(".txt")][0]
            print(f"  found inside zip: {name}")
            with z.open(name) as f:
                raw_bytes = f.read()
    else:
        raise FileNotFoundError(
            f"Could not find {path_txt} or {path_zip}. "
            "Download allwords-unique-sorted.txt.zip and place it here."
        )

    # detect and strip BOM, then decode
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1252"):
        try:
            text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise ValueError("Could not decode the word file with any known encoding.")

    # debug: show first 3 raw lines so we can see what's in the file
    sample_lines = text.splitlines()[:5]
    print(f"  encoding detected, first 5 raw lines: {sample_lines}")
    print(f"  total lines in file: {len(text.splitlines()):,}")

    words = []
    for line in text.splitlines():
        # handle "word\tfrequency" or "word,frequency" or plain "word"
        parts = line.strip().split("\t")
        w = parts[0].strip().lower()
        if (MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN) and GEORGIAN_RE.match(w):
            words.append(w)

    print(f"  lines passing Georgian filter: {len(words):,}")

    # file is sorted by frequency descending — preserve that order so we
    # keep the most common/useful words when we apply max_words cap
    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    print(f"  unique words after dedup: {len(unique):,}")

    if max_words:
        unique = unique[:max_words]   # top-N by frequency

    print(f"  → {len(unique):,} words kept (max_words={max_words})")
    return unique


words = load_words(RAW_WORDS_PATH, RAW_WORDS_ZIP, max_words=MAX_WORDS)

if len(words) == 0:
    raise RuntimeError(
        "No Georgian words were loaded. Check the debug output above:\n"
        "  - Are the 'first 5 raw lines' actual Georgian text?\n"
        "  - If they look like garbled bytes, try changing the encoding in load_words.\n"
        "  - If lines are empty, the file may be structured differently (e.g. one word per line vs CSV)."
    )


# Full Georgian alphabet — used for random substitutions
GEO_ALPHA = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")

# Georgian keyboard layout rows (QWERTY-style mapping)
# Each list = characters whose keys are physically adjacent on a Georgian keyboard
GEO_KEYBOARD_ROWS = [
    list("ქწერტყუიოპ"),
    list("ასდფგჰჯკლ"),
    list("ზხცვბნმ"),
]

# Build adjacency map: char → list of physically nearby chars
GEO_ADJACENCY: dict[str, list[str]] = {}
for row in GEO_KEYBOARD_ROWS:
    for i, ch in enumerate(row):
        neighbors = []
        if i > 0:
            neighbors.append(row[i - 1])
        if i < len(row) - 1:
            neighbors.append(row[i + 1])
        GEO_ADJACENCY[ch] = neighbors


def _adjacent_or_random(ch: str) -> str:
    candidates = GEO_ADJACENCY.get(ch, [])
    if candidates:
        return random.choice(candidates)
    return random.choice(GEO_ALPHA)


def corrupt_word(word: str) -> str:
    if len(word) < 2:
        return word   # too short to corrupt meaningfully

    ops = ["substitute", "delete", "insert", "transpose"]
    # weight substitution/transpose higher — they're most common typos
    weights = [0.40, 0.25, 0.15, 0.20]
    op = random.choices(ops, weights=weights, k=1)[0]

    w = list(word)
    i = random.randint(0, len(w) - 1)

    if op == "substitute":
        w[i] = _adjacent_or_random(w[i])

    elif op == "delete":
        w.pop(i)

    elif op == "insert":
        w.insert(i, random.choice(GEO_ALPHA))

    elif op == "transpose" and i < len(w) - 1:
        w[i], w[i + 1] = w[i + 1], w[i]

    corrupted = "".join(w)
    # make sure we actually changed something (e.g. substitute chose same char)
    if corrupted == word:
        corrupted = corrupt_word(word)   # retry once
    return corrupted


def build_pairs(words, errors_per_word=2):
    pairs = []
    for word in words:
        pairs.append((word, word))                          # clean copy
        for _ in range(errors_per_word):
            pairs.append((corrupt_word(word), word))        # corrupted copy
    random.shuffle(pairs)
    return pairs


print("\nGenerating training pairs …")
all_pairs = build_pairs(words, errors_per_word=ERRORS_PER_WORD)
print(f"  → {len(all_pairs):,} pairs total")

# Show a few examples
print("\nSample pairs  (input → target):")
for inp, tgt in random.sample(all_pairs, 8):
    marker = "✓" if inp == tgt else "✗"
    print(f"  {marker}  {inp:20s} → {tgt}")


SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def build_vocab(pairs):
    chars = set()
    for inp, tgt in pairs:
        chars.update(inp)
        chars.update(tgt)
    vocab = SPECIAL_TOKENS + sorted(chars)
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char  = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char

char2idx, idx2char = build_vocab(all_pairs)
print(f"\nVocabulary size : {len(char2idx)}")
print(f"Chars           : {''.join(sorted(c for c in char2idx if c not in SPECIAL_TOKENS))}")

PAD_IDX = char2idx["<pad>"]
SOS_IDX = char2idx["<sos>"]
EOS_IDX = char2idx["<eos>"]
UNK_IDX = char2idx["<unk>"]
VOCAB_SIZE = len(char2idx)

def encode(word, char2idx):
    """word → list of indices with EOS appended"""
    return [char2idx.get(c, UNK_IDX) for c in word] + [EOS_IDX]

def encode_target(word, char2idx):
    """target → SOS + indices + EOS  (decoder needs SOS at position 0)"""
    return [SOS_IDX] + [char2idx.get(c, UNK_IDX) for c in word] + [EOS_IDX]


encoded_pairs = [(encode(inp, char2idx), encode_target(tgt, char2idx))
                 for inp, tgt in all_pairs]

max_input_len  = max(len(p[0]) for p in encoded_pairs)
max_target_len = max(len(p[1]) for p in encoded_pairs)
print(f"\nMax input length  : {max_input_len}")
print(f"Max target length : {max_target_len}")

# train / val split
random.shuffle(encoded_pairs)
split       = int((1 - VAL_RATIO) * len(encoded_pairs))
train_data  = encoded_pairs[:split]
val_data    = encoded_pairs[split:]
print(f"Train pairs : {len(train_data):,}")
print(f"Val   pairs : {len(val_data):,}")


class GeorgianSpellDataset(Dataset):
    def __init__(self, data, max_input_len, max_target_len):
        self.data           = data
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

    def _pad(self, seq, max_len):
        return seq + [PAD_IDX] * (max_len - len(seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, tgt = self.data[idx]
        return (
            torch.tensor(self._pad(inp, self.max_input_len),  dtype=torch.long),
            torch.tensor(self._pad(tgt, self.max_target_len), dtype=torch.long),
        )


train_dataset = GeorgianSpellDataset(train_data, max_input_len, max_target_len)
val_dataset   = GeorgianSpellDataset(val_data,   max_input_len, max_target_len)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# sanity-check one batch
src_batch, tgt_batch = next(iter(train_loader))
print(f"\nBatch shapes — src: {src_batch.shape}, tgt: {tgt_batch.shape}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru        = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)
        return outputs, hidden      # return ALL timesteps for attention


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch, src_len, _ = encoder_outputs.shape
        h = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([h, encoder_outputs], dim=2)))
        attn_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention  = Attention(hidden_dim)
        self.gru        = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc         = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        emb     = self.embedding(x)
        context = self.attention(hidden, encoder_outputs)
        rnn_in  = torch.cat([emb, context], dim=2)
        output, hidden = self.gru(rnn_in, hidden)
        logits  = self.fc(output)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size  = src.size(0)
        trg_len     = trg.size(1)
        vocab_size  = self.decoder.fc.out_features

        outputs          = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        enc_out, hidden  = self.encoder(src)

        input_char = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            logits, hidden = self.decoder(input_char, hidden, enc_out)
            outputs[:, t]  = logits.squeeze(1)

            use_teacher = random.random() < teacher_forcing_ratio
            top1        = logits.argmax(2)
            input_char  = trg[:, t].unsqueeze(1) if use_teacher else top1

        return outputs


encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, PAD_IDX)
decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, PAD_IDX)
model   = Seq2Seq(encoder, decoder).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


def run_epoch(model, loader, optimizer, criterion, tf_ratio, train=True):
    model.train() if train else model.eval()
    total_loss = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, trg in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            if train:
                optimizer.zero_grad()

            output = model(src, trg, tf_ratio if train else 0.0)
            # output: (batch, trg_len, vocab)  — skip position 0 (<sos>)
            output_flat = output[:, 1:].reshape(-1, VOCAB_SIZE)
            trg_flat    = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, trg_flat)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


train_losses, val_losses = [], []
best_val_loss = float("inf")
epochs_no_improve = 0
tf = TEACHER_FORCING

print("\nTraining:")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'TF':>5}  {'LR':>8}")
print("─" * 52)

for epoch in range(1, N_EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, optimizer, criterion, tf, train=True)
    val_loss   = run_epoch(model, val_loader,   optimizer, criterion, tf, train=False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    tf = max(TF_MIN, tf * TF_DECAY)

    flag = ""
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_spellchecker.pt")
        flag = "  ← best"
    else:
        epochs_no_improve += 1

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  {tf:>5.2f}  {current_lr:>8.6f}{flag}")

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses,   label="Val loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
plt.title("Georgian Spellchecker — Training Curves")
plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=120)
plt.show()
print(f"Loss curve saved to {OUTPUT_PLOT}")



# reload best weights before saving
model.load_state_dict(torch.load("best_spellchecker.pt", map_location="cpu"))

torch.save({
    "model_state"    : model.state_dict(),
    "char2idx"       : char2idx,
    "idx2char"       : idx2char,
    "max_input_len"  : max_input_len,
    "max_target_len" : max_target_len,
    "vocab_size"     : VOCAB_SIZE,
    "embed_dim"      : EMBED_DIM,
    "hidden_dim"     : HIDDEN_DIM,
}, OUTPUT_MODEL)

print(f"\n✓ Full checkpoint saved to {OUTPUT_MODEL}")
print(f"  vocab_size     : {VOCAB_SIZE}")
print(f"  best_val_loss  : {best_val_loss:.4f}")