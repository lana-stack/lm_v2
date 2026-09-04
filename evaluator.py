import os
import math
from datetime import datetime

import torch

from config import (
    DEVICE,
    TEMPERATURE,
    GENERATE_TOKENS,
    SAMPLES_DIR,
    BLOCK_SIZE
)


@torch.no_grad()
def evaluate_model(
    model,
    data,
    get_batch,
    batches=50
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

    loss = (
        sum(losses)
        /
        len(losses)
    )

    perplexity = math.exp(
        min(loss, 20)
    )

    return loss, perplexity


@torch.no_grad()
def generate_sample(
    model,
    text,
    stoi,
    itos
):

    model.eval()

    encoded = [
        stoi[ch]
        for ch in text[-BLOCK_SIZE:]
        if ch in stoi
    ]

    if not encoded:
        return ""

    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    generated = model.generate(
        context,
        GENERATE_TOKENS,
        TEMPERATURE
    )

    result = "".join(
        itos[i]
        for i in generated[0].tolist()
    )

    return result


def save_sample(
    sample,
    session
):

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    path = os.path.join(
        SAMPLES_DIR,
        f"session_{session}_{timestamp}.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(sample)

    return path
