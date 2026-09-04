import os
import json
import time

import torch

from config import (
    DEVICE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    BLOCK_SIZE,
    MODEL_PATH,
    STATE_PATH,
    CHECKPOINT_DIR,
    COLLECT_EVERY_SESSIONS,
    SAVE_EVERY_SESSION
)

from model import (
    GPTLanguageModel,
    expand_vocabulary
)

from trainer import (
    load_corpus,
    build_tokenizer,
    create_data,
    get_batch,
    train_session
)

from evaluator import (
    evaluate_model,
    generate_sample,
    save_sample
)

from collector import collect


# ============================================================
# STATE
# ============================================================

def load_state():

    if not os.path.exists(
        STATE_PATH
    ):

        return {
            "sessions": 0,
            "best_val_loss": None
        }

    try:

        with open(
            STATE_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {
            "sessions": 0,
            "best_val_loss": None
        }


def save_state(state):

    with open(
        STATE_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    session,
    val_loss
):

    path = os.path.join(
        CHECKPOINT_DIR,
        f"session_{session}.pth"
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "session":
                session,

            "val_loss":
                val_loss,

            "vocab_size":
                model.vocab_size,

            "n_embd":
                N_EMBD,

            "n_head":
                N_HEAD,

            "n_layer":
                N_LAYER,

            "block_size":
                BLOCK_SIZE
        },
        path
    )

    print(
        "Checkpoint:",
        path
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("MIND V1 — CONTINUOUS LEARNING")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

print(
    "Embedding:",
    N_EMBD
)

print(
    "Layers:",
    N_LAYER
)

print(
    "Heads:",
    N_HEAD
)

print(
    "Context:",
    BLOCK_SIZE
)

print("=" * 70)


# ============================================================
# LOAD CORPUS
# ============================================================

print()
print("Loading corpus...")

text = load_corpus()

chars, stoi, itos = (
    build_tokenizer(text)
)

vocab_size = len(chars)

print(
    "Vocabulary:",
    vocab_size
)

data = create_data(
    text,
    stoi
)

print(
    "Corpus characters:",
    len(data)
)


# ============================================================
# MODEL
# ============================================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE
).to(DEVICE)


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

state = load_state()

start_session = state.get(
    "sessions",
    0
)


if os.path.exists(
    MODEL_PATH
):

    print()
    print(
        "Loading existing model..."
    )

    saved_state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Determine vocabulary size stored in the old model.
    # --------------------------------------------------------

    saved_vocab_size = (
        saved_state[
            "token_embedding_table.weight"
        ].shape[0]
    )

    print(
        "Saved model vocabulary:",
        saved_vocab_size
    )

    print(
        "Current vocabulary:",
        vocab_size
    )

    # --------------------------------------------------------
    # If vocabulary grew, first create a model with the
    # OLD vocabulary, load the old weights, then expand it.
    # --------------------------------------------------------

    if vocab_size > saved_vocab_size:

        print()
        print(
            "Vocabulary has grown."
        )

        print(
            f"Expanding model: "
            f"{saved_vocab_size} -> {vocab_size}"
        )

        old_model = GPTLanguageModel(
            vocab_size=saved_vocab_size,
            n_embd=N_EMBD,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            block_size=BLOCK_SIZE
        ).to(DEVICE)

        old_model.load_state_dict(
            saved_state
        )

        model = expand_vocabulary(
            old_model,
            vocab_size
        )

        print()
        print(
            "Old model weights preserved."
        )

        print(
            "New vocabulary weights initialized."
        )

    elif vocab_size == saved_vocab_size:

        model.load_state_dict(
            saved_state
        )

        print(
            "Vocabulary unchanged."
        )

    else:

        print()
        print(
            "ERROR:"
        )

        print(
            "Current vocabulary is smaller "
            "than the saved model vocabulary."
        )

        print(
            f"Saved: {saved_vocab_size}"
        )

        print(
            f"Current: {vocab_size}"
        )

        print(
            "Stopping to protect the model."
        )

        raise RuntimeError(
            "Vocabulary shrink detected."
        )

else:

    print()
    print(
        "No model found."
    )

    print(
        "Starting from scratch."
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# ============================================================
# LOOP
# ============================================================

session = start_session

try:

    while True:

        session += 1

        print()
        print("=" * 70)
        print(
            f"TRAINING SESSION {session}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Rebuild corpus
        # ----------------------------------------------------

        text = load_corpus()

        chars, stoi, itos = (
            build_tokenizer(text)
        )

        new_vocab_size = len(chars)

        # ----------------------------------------------------
        # Detect vocabulary growth after startup.
        # ----------------------------------------------------

        if new_vocab_size > model.vocab_size:

            old_vocab_size = model.vocab_size

            model = expand_vocabulary(
                model,
                new_vocab_size
            )

            # ------------------------------------------------
            # Recreate optimizer because model parameters
            # have changed.
            # ------------------------------------------------

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4
            )

            print(
                f"Optimizer rebuilt for vocabulary "
                f"{old_vocab_size} -> {new_vocab_size}"
            )

        elif new_vocab_size < model.vocab_size:

            print()
            print(
                "ERROR:"
            )

            print(
                "Vocabulary became smaller."
            )

            print(
                "Stopping to protect the model."
            )

            break

        data = create_data(
            text,
            stoi
        )

        print(
            "Vocabulary:",
            model.vocab_size
        )

        print(
            "Corpus characters:",
            len(data)
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        split = int(
            len(data) * 0.9
        )

        train_data = data[
            :split
        ]

        val_data = data[
            split:
        ]

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = train_session(
            model,
            train_data,
            optimizer
        )

        print()
        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        def validation_batch(
            *args,
            **kwargs
        ):

            return get_batch(
                val_data
            )

        val_loss, perplexity = (
            evaluate_model(
                model,
                val_data,
                validation_batch
            )
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Perplexity: "
            f"{perplexity:.4f}"
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        prompt = text[
            max(
                0,
                len(text) - BLOCK_SIZE
            ):
        ]

        sample = generate_sample(
            model,
            prompt,
            stoi,
            itos
        )

        sample_path = save_sample(
            sample,
            session
        )

        print()
        print(
            "GENERATED SAMPLE"
        )

        print(
            "-" * 70
        )

        print(
            sample
        )

        print(
            "-" * 70
        )

        print(
            "Sample saved:",
            sample_path
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        if SAVE_EVERY_SESSION:

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Model saved:",
                MODEL_PATH
            )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            session,
            val_loss
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        best = state.get(
            "best_val_loss"
        )

        if (
            best is None
            or val_loss < best
        ):

            state[
                "best_val_loss"
            ] = val_loss

            print(
                "NEW BEST VALIDATION LOSS"
            )

        state[
            "sessions"
        ] = session

        state[
            "vocab_size"
        ] = model.vocab_size

        save_state(
            state
        )

        # ----------------------------------------------------
        # INTERNET COLLECTION
        # ----------------------------------------------------

        if (
            session
            %
            COLLECT_EVERY_SESSIONS
            ==
            0
        ):

            collect()

            print()
            print(
                "New material collected."
            )

        # ----------------------------------------------------
        # CONTINUE
        # ----------------------------------------------------

        print()
        print(
            "Continuing training..."
        )

        if (
            session
            %
            10
            ==
            0
        ):

            print(
                "Session milestone:",
                session
            )

        if session > 0:

            time.sleep(1)


except KeyboardInterrupt:

    print()
    print("=" * 70)
    print("STOPPING MIND")
    print("=" * 70)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    state[
        "sessions"
    ] = session

    state[
        "vocab_size"
    ] = model.vocab_size

    save_state(
        state
    )

    print(
        "Model saved."
    )

    print(
        "State saved."
    )

    print(
        "Safe shutdown complete."
    )