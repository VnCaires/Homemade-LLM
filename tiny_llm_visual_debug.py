import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_bpe_tokenizer import SimpleBPETokenizer
from training_text import TRAINING_TEXT


# ============================================================
# 1. YOUR TRAINING TEXT
# ============================================================
# Edit `training_text.py` to change the corpus used by the model.

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. CONFIG
# ============================================================
@dataclass
class Config:
    # Larger default config tuned for the current tokenized corpus and RTX 4060-class GPUs.
    batch_size: int = 32
    block_size: int = 128         # how many tokens the model can look at once
    max_steps: int = 4000
    eval_interval: int = 200
    learning_rate: float = 1e-3

    n_embed: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    tokenizer_vocab_size: int = 1024
    tokenizer_min_pair_frequency: int = 3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ============================================================
# 3. TOKENIZER (byte-level BPE)
# ============================================================
text = TRAINING_TEXT
tokenizer = SimpleBPETokenizer.train_or_load(
    text,
    target_vocab_size=cfg.tokenizer_vocab_size,
    min_pair_frequency=cfg.tokenizer_min_pair_frequency,
    model_path=Path(__file__).with_name("tokenizer_model.json"),
)
vocab_size = tokenizer.vocab_size

def encode(s: str):
    return tokenizer.encode(s)

def decode(ids):
    return tokenizer.decode(ids)

data = torch.tensor(encode(text), dtype=torch.long)


def prepare_splits(full_data: torch.Tensor):
    if len(full_data) < 4:
        raise ValueError("Training text is too short. Add a few more characters to study the model.")

    # Keep the educational config, but shrink block size when the corpus is tiny.
    max_supported_block = max(1, (len(full_data) - 2) // 2)
    if cfg.block_size > max_supported_block:
        print(
            f"Requested block_size={cfg.block_size} is too large for this corpus, "
            f"so it was reduced to {max_supported_block}."
        )
        cfg.block_size = max_supported_block

    min_split_size = cfg.block_size + 1
    proposed_train_size = int(0.9 * len(full_data))
    train_size = min(max(proposed_train_size, min_split_size), len(full_data) - min_split_size)

    train_split = full_data[:train_size]
    val_split = full_data[train_size:]
    return train_split, val_split


train_data, val_data = prepare_splits(data)


# ============================================================
# 4. DATA BATCHES
# ============================================================
def get_batch(split: str):
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")

    source = train_data if split == "train" else val_data
    max_start = len(source) - cfg.block_size - 1
    ix = torch.randint(0, max_start + 1, (cfg.batch_size,))
    x = torch.stack([source[i:i + cfg.block_size] for i in ix])
    y = torch.stack([source[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# ============================================================
# 5. TRANSFORMER PARTS
# ============================================================
class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )

        self.last_attention = None

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.last_attention = wei.detach().cpu()
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embed, 4 * cfg.n_embed),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embed, cfg.n_embed),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = cfg.n_embed // cfg.n_heads
        self.sa = MultiHeadAttention(cfg.n_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=cfg.device))  # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last time step
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


model = TinyTransformerLM().to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)


def choose_study_prompt(preferred: str = "Call me"):
    if preferred and preferred in text:
        return preferred
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[: min(20, len(first_line))] if first_line else "The"


def choose_demo_prompts():
    candidates = ["Call me", "the ", "whale", "ship", "sea"]
    prompts = [candidate for candidate in candidates if candidate in text]
    return prompts[:3] if prompts else [choose_study_prompt()]


def print_tensor_shape(name: str, tensor: torch.Tensor):
    print(f"{name:<28} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")


def printable_token_preview(max_tokens: int = 40):
    preview = [tokenizer.token_to_display(token_id) for token_id in range(min(max_tokens, vocab_size))]
    if vocab_size > max_tokens:
        preview.append("...")
    return preview


def printable_token(token_id: int, max_length: int = 16):
    token_text = tokenizer.token_to_display(token_id)
    if len(token_text) <= max_length:
        return token_text
    return token_text[: max_length - 3] + "..."


def printable_text(text_value: str):
    return text_value.encode("unicode_escape").decode("ascii")


@torch.no_grad()
def sanity_check(prompt: str = "Call me"):
    print("\n" + "=" * 60)
    print("SANITY CHECK: one forward-pass walkthrough")
    print("=" * 60)

    xb, yb = get_batch("train")
    print_tensor_shape("input batch xb", xb)
    print_tensor_shape("target batch yb", yb)

    idx = xb[:1]
    print_tensor_shape("single example idx", idx)
    sample_tokens = [printable_token(token_id) for token_id in idx[0, : min(12, idx.shape[1])].tolist()]
    print(f"sample input tokens         {sample_tokens}")

    tok_emb = model.token_embedding_table(idx)
    pos_idx = torch.arange(idx.shape[1], device=cfg.device)
    pos_emb = model.position_embedding_table(pos_idx)
    print_tensor_shape("token embeddings", tok_emb)
    print_tensor_shape("position indices", pos_idx)
    print_tensor_shape("position embeddings", pos_emb)

    x = tok_emb + pos_emb
    print_tensor_shape("embedding sum", x)

    first_block = model.blocks[0]
    ln1_out = first_block.ln1(x)
    print_tensor_shape("after layer norm 1", ln1_out)

    first_head = first_block.sa.heads[0]
    head_out = first_head(ln1_out)
    print_tensor_shape("first attention head out", head_out)
    if first_head.last_attention is not None:
        print_tensor_shape("saved attention matrix", first_head.last_attention)

    logits, loss = model(xb, yb)
    print_tensor_shape("model logits", logits)
    if loss is not None:
        print(f"example loss value           {loss.item():.4f}")

    safe_prompt = choose_study_prompt(prompt)
    print(f"debug prompt used            {repr(safe_prompt)}")
    debug_next_token(safe_prompt, top_k=8, plot=False)


# ============================================================
# 6. EVALUATION
# ============================================================
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(20)
        for k in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================
# 7. VISUAL DEBUG: NEXT-TOKEN PROBABILITIES
# ============================================================
@torch.no_grad()
def debug_next_token(prompt: str, top_k: int = 10, plot: bool = True):
    model.eval()

    prompt_ids = encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt cannot be empty.")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    idx_cond = idx[:, -cfg.block_size:]

    logits, _ = model(idx_cond)
    last_logits = logits[0, -1]
    probs = F.softmax(last_logits, dim=-1).detach().cpu()

    top_probs, top_idx = torch.topk(probs, k=min(top_k, vocab_size))

    print("\n" + "=" * 60)
    print(f"PROMPT: {repr(prompt)}")
    print("Top next-token predictions:")
    for rank, (p, i) in enumerate(zip(top_probs.tolist(), top_idx.tolist()), start=1):
        shown = printable_token(i)
        print(f"{rank:2d}. {repr(shown):>12} -> {p * 100:6.2f}%")
    print("=" * 60)

    if plot:
        labels = []
        values = []
        for p, i in zip(top_probs.tolist(), top_idx.tolist()):
            labels.append(printable_token(i, max_length=10))
            values.append(p * 100)

        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.title(f"Next-token probabilities for prompt: {repr(prompt)}")
        plt.xlabel("Token")
        plt.ylabel("Probability (%)")
        plt.tight_layout()
        plt.show()

    model.train()


# ============================================================
# 8. VISUAL DEBUG: ATTENTION HEATMAP
# ============================================================
@torch.no_grad()
def show_attention(prompt: str, layer_index: int = 0, head_index: int = 0):
    model.eval()

    prompt_ids = encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt cannot be empty.")
    if not 0 <= layer_index < len(model.blocks):
        raise ValueError(f"layer_index must be between 0 and {len(model.blocks) - 1}.")

    block = model.blocks[layer_index]
    if not 0 <= head_index < len(block.sa.heads):
        raise ValueError(f"head_index must be between 0 and {len(block.sa.heads) - 1}.")

    prompt_ids = prompt_ids[-cfg.block_size:]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)
    _logits, _ = model(idx)

    attn = block.sa.heads[head_index].last_attention  # shape: (B, T, T)
    if attn is None:
        print("No attention saved.")
        return

    attn = attn[0].numpy()
    token_labels = [printable_token(token_id, max_length=8) for token_id in prompt_ids[-attn.shape[0]:]]

    plt.figure(figsize=(8, 6))
    plt.imshow(attn)
    plt.colorbar()
    plt.title(f"Attention heatmap - layer {layer_index}, head {head_index}")
    plt.xticks(range(len(token_labels)), token_labels, rotation=90)
    plt.yticks(range(len(token_labels)), token_labels)
    plt.xlabel("Looks at")
    plt.ylabel("Current position")
    plt.tight_layout()
    plt.show()

    model.train()


def run_training(max_steps: int | None = None, debug_shapes: bool = False):
    train_losses = []
    val_losses = []
    steps_seen = []
    total_steps = cfg.max_steps if max_steps is None else max_steps

    print(f"Device: {cfg.device}")
    print("Tokenizer: byte-level BPE")
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Tokenizer merges: {tokenizer.num_merges}")
    print(f"Token preview: {printable_token_preview()}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Using block size: {cfg.block_size}")
    print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")
    print(f"Corpus characters: {len(text)} | Corpus tokens: {len(data)}")
    print(f"Training steps: {total_steps}")

    if debug_shapes:
        sanity_check()

    for step in range(total_steps):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == total_steps - 1:
            losses = estimate_loss()
            steps_seen.append(step)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            # Show how the model currently thinks.
            debug_prompt = choose_study_prompt("Call me")
            debug_next_token(debug_prompt, top_k=8, plot=False)

    return steps_seen, train_losses, val_losses


def plot_losses(steps_seen, train_losses, val_losses, enabled: bool = True):
    if not enabled:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(steps_seen, train_losses, label="train loss")
    plt.plot(steps_seen, val_losses, label="val loss")
    plt.title("Training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_final_examples(show_plots: bool = True):
    start_prompt = choose_study_prompt("Call me")
    context = torch.tensor([encode(start_prompt)], dtype=torch.long, device=cfg.device)
    generated = model.generate(context, max_new_tokens=200)[0].tolist()

    print("\n" + "#" * 60)
    print("GENERATED TEXT")
    print("#" * 60)
    print(f"Seed prompt: {repr(start_prompt)}")
    print(printable_text(decode(generated)))

    for prompt in choose_demo_prompts():
        debug_next_token(prompt, top_k=10, plot=show_plots)
    if show_plots:
        show_attention(start_prompt, layer_index=0, head_index=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and inspect a tiny token-level transformer for study."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the number of training steps for short experiments.",
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help="Print a study-friendly walkthrough of tensor shapes before training.",
    )
    parser.add_argument(
        "--debug-only",
        action="store_true",
        help="Run the tensor-shape walkthrough and next-token debug without training.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib windows. Useful for quick runs and CI.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be at least 1.")

    if args.debug_only:
        sanity_check()
        return

    steps_seen, train_losses, val_losses = run_training(
        max_steps=args.steps,
        debug_shapes=args.debug_shapes,
    )
    plot_losses(steps_seen, train_losses, val_losses, enabled=not args.skip_plots)
    show_final_examples(show_plots=not args.skip_plots)


if __name__ == "__main__":
    main()
