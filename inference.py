import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch, src_len, _ = encoder_outputs.shape
        h = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([h, encoder_outputs], dim=2)))
        attn_weights = F.softmax(self.v(energy).squeeze(2), dim=1)
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

    def forward(self, src, trg, teacher_forcing_ratio=0.0):
        batch_size       = src.size(0)
        trg_len          = trg.size(1)
        vocab_size       = self.decoder.fc.out_features
        outputs          = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        enc_out, hidden  = self.encoder(src)
        input_char       = trg[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            logits, hidden = self.decoder(input_char, hidden, enc_out)
            outputs[:, t]  = logits.squeeze(1)
            input_char     = logits.argmax(2)
        return outputs



_cache: dict = {}

def _load(model_path: str):
    if model_path in _cache:
        return _cache[model_path]

    checkpoint     = torch.load(model_path, map_location="cpu", weights_only=False)
    char2idx       = checkpoint["char2idx"]
    idx2char       = checkpoint["idx2char"]
    max_input_len  = checkpoint["max_input_len"]
    max_target_len = checkpoint["max_target_len"]
    vocab_size     = checkpoint["vocab_size"]
    embed_dim      = checkpoint["embed_dim"]
    hidden_dim     = checkpoint["hidden_dim"]
    pad_idx        = char2idx["<pad>"]

    enc   = Encoder(vocab_size, embed_dim, hidden_dim, pad_idx)
    dec   = Decoder(vocab_size, embed_dim, hidden_dim, pad_idx)
    model = Seq2Seq(enc, dec)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _cache[model_path] = (model, char2idx, idx2char, max_input_len, max_target_len)
    return _cache[model_path]

def correct_word(word: str, model_path: str) -> str:
    model, char2idx, idx2char, max_input_len, max_target_len = _load(model_path)

    device  = next(model.parameters()).device
    pad_idx = char2idx["<pad>"]
    sos_idx = char2idx["<sos>"]
    eos_idx = char2idx["<eos>"]
    unk_idx = char2idx.get("<unk>", pad_idx)

    seq = [char2idx.get(c, unk_idx) for c in word] + [eos_idx]
    seq = (seq + [pad_idx] * max_input_len)[:max_input_len]

    src = torch.tensor([seq], dtype=torch.long).to(device)

    with torch.no_grad():
        enc_out, hidden = model.encoder(src)

    decoded    = []
    input_char = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

    for _ in range(max_target_len):
        with torch.no_grad():
            logits, hidden = model.decoder(input_char, hidden, enc_out)
        pred_idx   = logits.argmax(2).item()
        if pred_idx == eos_idx:
            break
        if pred_idx not in (sos_idx, pad_idx):
            decoded.append(idx2char[pred_idx])
        input_char = torch.tensor([[pred_idx]], dtype=torch.long).to(device)

    return "".join(decoded)


DEMO_EXAMPLES = [
    # (input_word,        expected_correct,   description)
    ("სახლი",             "სახლი",            "already correct"),
    ("წყალი",             "წყალი",            "already correct"),
    ("კაცი",              "კაცი",             "already correct"),
    ("ქალი",              "ქალი",             "already correct"),
    ("ბავშვი",            "ბავშვი",           "already correct"),
    ("სკოლა",             "სკოლა",            "already correct"),
    ("მასწავლებელი",      "მასწავლებელი",     "already correct"),
    ("საქართველო",        "საქართველო",       "already correct"),
    # ── substitution errors (one wrong char) ──
    ("სახხლი",             "სახლი",            "insertion: extra ხ"),
    ("კახუ", "კახი", "substitution: უ→ი"),
    ("მაგიდს", "მაგიდა", "substitution: final ს→ა"),

    ("ქალლი", "ქალი", "insertion: extra ლ"),
    ("წყაკი",             "წყალი",            "substitution: კ→ლ"),
    # ── deletion errors (one char missing) ──
    ("სკოა",              "სკოლა",            "deletion: ლ missing"),
    ("ბაშვვი",             "ბავშვი",           "swapped letters : შ & ვ"),
    ("საქრთველო",         "საქართველო",       "deletion: ა missing"),
    ("მასწავლბელი",       "მასწავლებელი",     "deletion: ე missing"),
    # ── transposition errors (two chars swapped) ──
    ("სახლი",             "სახლი",            "already correct (control)"),
    ("წაყლი",             "წყალი",            "transposition: ა↔ყ"),
    ("კაცი",              "კაცი",             "already correct (control)"),
    ("ბავვში",            "ბავშვი",           "transposition: შ↔ვ"),
    # ── insertion errors (one extra char) ──
    ("სკოოლა",            "სკოლა",            "insertion: extra ო"),
    # ── substitution — one wrong character ──
    ("მაგიდს",         "მაგიდა",         "substitution: final ს→ა"),
    ("ტელეფონუ",       "ტელეფონი",       "substitution: final უ→ი"),
    ("ვადრუ",          "ვადრი",          "substitution: final უ→ი"),
    ("წყაკი",          "წყალი",          "substitution: კ→ლ"),
    # ── deletion — one char missing ──
    ("მასწავლბელი",    "მასწავლებელი",   "deletion: ე missing"),
    # ── insertion — one extra char ──
    ("ბალიიში",        "ბალიში",         "insertion: extra ი fixed"),
    ("სახხე",          "სახე",           "insertion: extra ხ"),
    ("ქაალი",          "ქალი",           "insertion: extra ა"),
]



def evaluate_model(model_path: str, words_file: str, n_words: int = 500):
    import re, random, zipfile, os

    GEORGIAN_RE = re.compile(r'^\[\u10D0-\u10FF\]+$')
    GEO_ALPHA = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
    GEO_KEYBOARD_ROWS = [list("ქწერტყუიოპ"), list("ასდფგჰჯკლ"), list("ზხცვბნმ")]
    GEO_ADJACENCY: dict = {}
    for row in GEO_KEYBOARD_ROWS:
        for i, ch in enumerate(row):
            neighbors = []
            if i > 0:            neighbors.append(row[i-1])
            if i < len(row)-1:   neighbors.append(row[i+1])
            GEO_ADJACENCY[ch] = neighbors

    def adjacent_or_random(ch):
        c = GEO_ADJACENCY.get(ch, [])
        return random.choice(c) if c else random.choice(GEO_ALPHA)

    def make_corruption(word, op):
        w = list(word)
        i = random.randint(0, len(w)-1)
        if op == "substitute":
            r = adjacent_or_random(w[i])
            if r == w[i]: return None
            w[i] = r
        elif op == "delete":
            if len(w) < 3: return None
            w.pop(i)
        elif op == "insert":
            w.insert(i, random.choice(GEO_ALPHA))
        elif op == "transpose":
            if i >= len(w)-1: return None
            w[i], w[i+1] = w[i+1], w[i]
        corrupted = "".join(w)
        return corrupted if corrupted != word else None

    # load words
    raw_bytes = None
    for path in (words_file, words_file + ".zip"):
        if os.path.exists(path):
            if path.endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    name = [n for n in z.namelist() if n.endswith(".txt")][0]
                    raw_bytes = z.open(name).read()
            else:
                raw_bytes = open(path, "rb").read()
            break
    if raw_bytes is None:
        print(f"  Could not find {words_file} — skipping evaluation.")
        return

    text = raw_bytes.decode("utf-8-sig", errors="replace")
    all_words = []
    for line in text.splitlines():
        w = line.strip().split("\t")[0].strip().lower()
        if 3 <= len(w) <= 18 and GEORGIAN_RE.match(w):
            all_words.append(w)

    # use words from position 35000+ so they're outside the training set
    test_pool = list(dict.fromkeys(all_words))[35_000:35_000 + n_words * 4]
    if not test_pool:
        print("  Not enough words beyond training range — using last 2000 words instead.")
        test_pool = list(dict.fromkeys(all_words))[-2000:]

    random.seed(99)   # fixed seed for reproducibility
    random.shuffle(test_pool)
    test_words = test_pool[:n_words]

    ops   = ["clean", "substitute", "delete", "insert", "transpose"]
    cases = {op: {"correct": 0, "total": 0} for op in ops}
    failures = []

    print(f"\nEvaluating on {len(test_words)} test words …")

    for word in test_words:
        op = random.choice(ops)
        if op == "clean":
            corrupted = word
        else:
            corrupted = make_corruption(word, op)
            if corrupted is None:          # couldn't make this error type
                corrupted = word
                op = "clean"

        predicted = correct_word(corrupted, model_path)
        correct   = predicted == word

        cases[op]["total"]   += 1
        cases[op]["correct"] += int(correct)

        if not correct and len(failures) < 10:
            failures.append((op, corrupted, predicted, word))

    total_c = sum(v["correct"] for v in cases.values())
    total_t = sum(v["total"]   for v in cases.values())

    print(f"\n{'─'*55}")
    print(f"  Overall accuracy : {total_c}/{total_t}  ({100*total_c/total_t:.1f}%)")
    print(f"{'─'*55}")
    print(f"  {'Type':<14}  {'Correct':>7}  {'Total':>7}  {'Acc':>6}")
    print(f"  {'─'*40}")
    for op in ops:
        c, t = cases[op]["correct"], cases[op]["total"]
        acc  = 100*c/t if t else 0
        print(f"  {op:<14}  {c:>7}  {t:>7}  {acc:>5.1f}%")
    print(f"{'─'*55}")

    if failures:
        print(f"\n  Sample failures (type | input → predicted | expected):")
        for op, inp, pred, exp in failures:
            print(f"    [{op}]  {inp} → {pred}  (expected: {exp})")




def evaluate_model(model_path: str, words_file: str, n_words: int = 500):
    import re, random, zipfile, os

    GEORGIAN_RE = re.compile(r"^[ა-ჿ]+$")
    GEO_ALPHA = list("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
    GEO_KEYBOARD_ROWS = [list("ქწერტყუიოპ"), list("ასდფგჰჯკლ"), list("ზხცვბნმ")]
    GEO_ADJACENCY = {}
    for row in GEO_KEYBOARD_ROWS:
        for i, ch in enumerate(row):
            neighbors = []
            if i > 0:             neighbors.append(row[i-1])
            if i < len(row) - 1:  neighbors.append(row[i+1])
            GEO_ADJACENCY[ch] = neighbors

    def adjacent_or_random(ch):
        c = GEO_ADJACENCY.get(ch, [])
        return random.choice(c) if c else random.choice(GEO_ALPHA)

    def make_corruption(word, op):
        w = list(word)
        i = random.randint(0, len(w) - 1)
        if op == "substitute":
            r = adjacent_or_random(w[i])
            if r == w[i]: return None
            w[i] = r
        elif op == "delete":
            if len(w) < 3: return None
            w.pop(i)
        elif op == "insert":
            w.insert(i, random.choice(GEO_ALPHA))
        elif op == "transpose":
            if i >= len(w) - 1: return None
            w[i], w[i+1] = w[i+1], w[i]
        corrupted = "".join(w)
        return corrupted if corrupted != word else None

    raw_bytes = None
    for path in (words_file, words_file + ".zip"):
        if os.path.exists(path):
            if path.endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    name = [n for n in z.namelist() if n.endswith(".txt")][0]
                    raw_bytes = z.open(name).read()
            else:
                raw_bytes = open(path, "rb").read()
            break

    if raw_bytes is None:
        print(f"Could not find {words_file} — skipping evaluation.")
        return

    text = raw_bytes.decode("utf-8-sig", errors="replace")
    all_words = []
    for line in text.splitlines():
        w = line.strip().split("\t")[0].strip().lower()
        if 3 <= len(w) <= 18 and GEORGIAN_RE.match(w):
            all_words.append(w)

    seen, unique = set(), []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    test_pool = unique[30_000:30_000 + n_words * 4]
    if len(test_pool) < n_words:
        test_pool = unique[-n_words * 4:]

    random.seed(99)
    random.shuffle(test_pool)
    test_words = test_pool[:n_words]

    ops = ["clean", "substitute", "delete", "insert", "transpose"]
    cases = {op: {"correct": 0, "total": 0} for op in ops}
    failures = []

    print(f"\nEvaluating on {len(test_words)} unseen test words ...")

    for word in test_words:
        op = random.choice(ops)
        if op == "clean":
            corrupted = word
        else:
            corrupted = make_corruption(word, op)
            if corrupted is None:
                corrupted = word
                op = "clean"

        predicted = correct_word(corrupted, model_path)
        correct   = predicted == word
        cases[op]["total"]   += 1
        cases[op]["correct"] += int(correct)

        if not correct and len(failures) < 8:
            failures.append((op, corrupted, predicted, word))

    total_c = sum(v["correct"] for v in cases.values())
    total_t = sum(v["total"]   for v in cases.values())

    print(f"\n{'─'*56}")
    print(f"  Overall accuracy : {total_c}/{total_t}  ({100*total_c/total_t:.1f}%)")
    print(f"{'─'*56}")
    print(f"  {'Type':<14}  {'Correct':>7}  {'Total':>7}  {'Acc%':>6}")
    print(f"  {'─'*38}")
    for op in ops:
        c, t = cases[op]["correct"], cases[op]["total"]
        print(f"  {op:<14}  {c:>7}  {t:>7}  {(100*c/t if t else 0):>5.1f}%")
    print(f"{'─'*56}")

    if failures:
        print(f"\n  Sample failures:")
        for op, inp, pred, exp in failures:
            print(f"    [{op:11s}]  {inp:18s} -> {pred:18s}  (expected: {exp})")


def run_demo(model_path: str):
    print(f"\nLoading model from: {model_path}")
    _load(model_path)
    print(f"Model loaded. Running {len(DEMO_EXAMPLES)} examples...\n")

    header = f"{'#':>3}  {'Input':24}  {'Output':24}  {'Expected':24}  OK?  Note"
    print(header)
    print("─" * len(header))

    correct_count = 0
    for i, (inp, expected, note) in enumerate(DEMO_EXAMPLES, 1):
        output = correct_word(inp, model_path)
        ok     = output == expected
        correct_count += ok
        print(f"{i:>3}  {inp:24}  {output:24}  {expected:24}  {'✓' if ok else '✗'}    {note}")

    pct = 100 * correct_count / len(DEMO_EXAMPLES)
    print(f"\nAccuracy on demo set: {correct_count}/{len(DEMO_EXAMPLES)} ({pct:.0f}%)")


def interactive(model_path: str):
    print("\n── Interactive mode (quit to exit) ──")
    _load(model_path)
    while True:
        word = input("\nსიტყვა: ").strip()
        if word.lower() in ("quit", "exit", "q"):
            break
        if not word:
            continue
        result = correct_word(word, model_path)
        print(f"  {'✓ სწორია:' if word == result else '→'} {result}")


if __name__ == "__main__":
    import sys
    path       = sys.argv[1] if len(sys.argv) > 1 else "final_spellchecker_complete.pt"
    words_file = sys.argv[2] if len(sys.argv) > 2 else "allwords-unique-sorted.txt"

    run_demo(path)

    if input("\nRun systematic evaluation on 500 unseen words? [y/N] ").strip().lower() == "y":
        evaluate_model(path, words_file, n_words=500)

    if input("\nRun interactive mode? [y/N] ").strip().lower() == "y":
        interactive(path)