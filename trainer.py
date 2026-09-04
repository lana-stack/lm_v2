import os
import glob
import json
import torch

from config import (
    DEVICE,
    BLOCK_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    TRAIN_STEPS_PER_SESSION,
    INPUT_PATH,
    CORPUS_DIR
)


# ============================================================
# VOCABULARY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "vocab.json"
)


# ============================================================
# CORPUS
# ============================================================

def load_corpus():

    texts = []

    if os.path.exists(INPUT_PATH):

        with open(
            INPUT_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            texts.append(
                f.read()
            )

    for path in glob.glob(
        os.path.join(
            CORPUS_DIR,
            "**",
            "*.txt"
        ),
        recursive=True
    ):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                texts.append(
                    f.read()
                )

        except Exception:
            pass

    if not texts:

        raise RuntimeError(
            "Training corpus is empty."
        )

    return "\n\n".join(texts)


# ============================================================
# VOCABULARY MANAGEMENT
# ============================================================

def load_vocab():

    if not os.path.exists(VOCAB_PATH):

        return None

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    return vocab["chars"]


def save_vocab(chars):

    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "chars": chars
            },
            f,
            ensure_ascii=False,
            indent=2
        )


def build_tokenizer(text):

    # --------------------------------------------------------
    # Load existing vocabulary
    # --------------------------------------------------------

    existing_chars = load_vocab()

    if existing_chars is None:

        # First creation
        chars = sorted(
            list(
                set(text)
            )
        )

        save_vocab(chars)

        print(
            f"Vocabulary created: "
            f"{len(chars)} characters"
        )

    else:

        chars = list(
            existing_chars
        )

        existing_set = set(chars)

        # ----------------------------------------------------
        # Find genuinely new characters
        # ----------------------------------------------------

        new_chars = sorted(
            set(text) - existing_set
        )

        if new_chars:

            old_size = len(chars)

            # IMPORTANT:
            # New characters are ALWAYS appended.
            # Existing character IDs NEVER change.
            chars.extend(
                new_chars
            )

            save_vocab(chars)

            print(
                f"Vocabulary expanded: "
                f"{old_size} -> {len(chars)}"
            )

            print(
                f"New characters: "
                f"{len(new_chars)}"
            )

            print(
                "Added:"
            )

            print(
                repr(
                    "".join(new_chars)
                )
            )

        else:

            print(
                f"Vocabulary unchanged: "
                f"{len(chars)} characters"
            )

    # --------------------------------------------------------
    # Build mappings
    # --------------------------------------------------------

    stoi = {
        ch: i
        for i, ch in enumerate(chars)
    }

    itos = {
        i: ch
        for i, ch in enumerate(chars)
    }

    return (
        chars,
        stoi,
        itos
    )


# ============================================================
# ENCODING
# ============================================================

def encode(
    text,
    stoi
):

    return [
        stoi[ch]
        for ch in text
        if ch in stoi
    ]


# ============================================================
# DATASET
# ============================================================

def create_data(
    text,
    stoi
):

    data = torch.tensor(
        encode(
            text,
            stoi
        ),
        dtype=torch.long
    )

    return data


# ============================================================
# BATCH
# ============================================================

def get_batch(
    data
):

    if len(data) <= (
        BLOCK_SIZE + 1
    ):

        raise RuntimeError(
            "Corpus is too small."
        )

    ix = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            data[
                i:i + BLOCK_SIZE
            ]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:
                i + BLOCK_SIZE + 1
            ]
            for i in ix
        ]
    )

    return (
        x.to(DEVICE),
        y.to(DEVICE)
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data,
    batches=20
):

    model.eval()

    losses = []

    for _ in range(
        batches
    ):

        xb, yb = get_batch(
            data
        )

        _, loss = model(
            xb,
            yb
        )

        losses.append(
            loss.item()
        )

    return (
        sum(losses)
        /
        len(losses)
    )


# ============================================================
# TRAINING SESSION
# ============================================================

def train_session(
    model,
    train_data,
    optimizer
):

    model.train()

    losses = []

    for step in range(
        TRAIN_STEPS_PER_SESSION
    ):

        xb, yb = get_batch(
            train_data
        )

        _, loss = model(
            xb,
            yb
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        losses.append(
            loss.item()
        )

        if (
            step + 1
        ) % 100 == 0:

            print(
                f"  step "
                f"{step + 1}/"
                f"{TRAIN_STEPS_PER_SESSION}"
                f"  loss={loss.item():.4f}"
            )

    return (
        sum(losses)
        /
        len(losses)
    )