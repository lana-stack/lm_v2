import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):

    def __init__(
        self,
        n_embd,
        head_size,
        block_size
    ):

        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(
            -2,
            -1
        )

        wei *= k.shape[-1] ** -0.5

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        v = self.value(x)

        return wei @ v


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        block_size
    ):

        super().__init__()

        if n_embd % n_head != 0:

            raise ValueError(
                "n_embd must be divisible by n_head"
            )

        head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd,
                    head_size,
                    block_size
                )
                for _ in range(n_head)
            ]
        )

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

    def forward(self, x):

        out = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        return self.proj(out)


class FeedForward(nn.Module):

    def __init__(
        self,
        n_embd
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            )
        )

    def forward(self, x):

        return self.net(x)


class Block(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        block_size
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.attention = MultiHeadAttention(
            n_embd,
            n_head,
            block_size
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )

        self.ffwd = FeedForward(
            n_embd
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        n_head,
        n_layer,
        block_size
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head,
                    block_size
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        if T > self.block_size:

            raise ValueError(
                "Sequence exceeds block size"
            )

        tok_emb = self.token_embedding_table(
            idx
        )

        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=idx.device
            )
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.reshape(
                B * T,
                C
            )

            targets = targets.reshape(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8
    ):

        self.eval()

        for _ in range(
            max_new_tokens
        ):

            idx_cond = idx[
                :, -self.block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :, -1, :
            ]

            logits /= temperature

            probs = F.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                [
                    idx,
                    idx_next
                ],
                dim=1
            )

        return idx


# ============================================================
# VOCABULARY EXPANSION
# ============================================================

def expand_vocabulary(
    model,
    new_vocab_size
):

    old_vocab_size = model.vocab_size

    if new_vocab_size <= old_vocab_size:

        return model

    print(
        f"Expanding model vocabulary: "
        f"{old_vocab_size} -> {new_vocab_size}"
    )

    device = next(
        model.parameters()
    ).device

    dtype = next(
        model.parameters()
    ).dtype

    # --------------------------------------------------------
    # Token embedding
    # --------------------------------------------------------

    old_embedding = (
        model.token_embedding_table
    )

    new_embedding = nn.Embedding(
        new_vocab_size,
        model.n_embd,
        device=device,
        dtype=dtype
    )

    nn.init.normal_(
        new_embedding.weight,
        mean=0.0,
        std=0.02
    )

    with torch.no_grad():

        new_embedding.weight[
            :old_vocab_size
        ].copy_(
            old_embedding.weight
        )

    model.token_embedding_table = (
        new_embedding
    )

    # --------------------------------------------------------
    # Language-model output head
    # --------------------------------------------------------

    old_head = model.lm_head

    new_head = nn.Linear(
        model.n_embd,
        new_vocab_size,
        device=device,
        dtype=dtype
    )

    nn.init.normal_(
        new_head.weight,
        mean=0.0,
        std=0.02
    )

    if new_head.bias is not None:

        nn.init.zeros_(
            new_head.bias
        )

    with torch.no_grad():

        new_head.weight[
            :old_vocab_size
        ].copy_(
            old_head.weight
        )

        if (
            old_head.bias is not None
            and new_head.bias is not None
        ):

            new_head.bias[
                :old_vocab_size
            ].copy_(
                old_head.bias
            )

    model.lm_head = new_head

    # --------------------------------------------------------
    # Update model vocabulary size
    # --------------------------------------------------------

    model.vocab_size = new_vocab_size

    print(
        "Vocabulary expansion complete."
    )

    return model