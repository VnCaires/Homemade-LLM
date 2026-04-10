# Homemade LLM

A small transformer-based language model built for learning and experimentation.

This project is intentionally simple, slow, and readable so you can study how a tiny transformer is trained, how it predicts the next token, and how its internal tensors evolve during a forward pass.

## Purpose

This repository exists for study.

Instead of hiding the logic behind a large framework, it keeps the main pieces visible:
- tokenization
- prompt-to-token breakdown in the terminal
- batching
- self-attention
- training loss
- next-token probabilities
- attention visualization

## Setup

CPU setup:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
```

CUDA setup:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
```

## Run

Full training run:

```powershell
python tiny_llm_visual_debug.py
```

Short study run:

```powershell
python tiny_llm_visual_debug.py --steps 2 --debug-shapes --skip-plots
```

Train and save a checkpoint:

```powershell
python tiny_llm_visual_debug.py --steps 1000 --checkpoint-path checkpoints\study.pt
```

Resume from that checkpoint:

```powershell
python tiny_llm_visual_debug.py --resume --steps 1000 --checkpoint-path checkpoints\study.pt
```

Debug walkthrough only:

```powershell
python tiny_llm_visual_debug.py --debug-only
```

The debug output prints the prompt, how it is split into BPE tokens, the token ids used by the model, and the current next-token probabilities.

Tokenizer walkthrough only:

```powershell
python simple_bpe_tokenizer.py --prompt "Call me"
```

Export the full tokenizer vocabulary to a file:

```powershell
python simple_bpe_tokenizer.py --export-vocab-path all_tokens.txt
```

## Study Flow

1. Edit `training_text.txt` to choose the training corpus.
2. Run `--debug-only` to inspect tensor shapes before training.
3. Run `--steps 2 --debug-shapes --skip-plots` for a very short training pass with live prompt tokenization and next-token predictions at each evaluation point.
4. Run the full script when you want to watch learning and plots over time.

## Files

- `tiny_llm_visual_debug.py`: main model, training loop, and debug tools
- `simple_bpe_tokenizer.py`: local byte-level BPE tokenizer used to build token ids and print a prompt-to-token walkthrough
- `training_text.txt`: the training text used for the corpus
- `training_text.py`: tiny loader that reads `training_text.txt`
- `STUDY_NOTES.md`: a place to write your own explanations while learning
- `checkpoints/`: local training checkpoints created when you run or resume training

## Notes

- The model now uses a local byte-level BPE tokenizer, which is more efficient than pure character-level training while still being small enough to study.
- Tokenization is fixed once the tokenizer is loaded or trained; the study output shows that tokenization live for each debug prompt while the model's probabilities change during training.
- The code is intentionally not optimized for performance. Readability comes first.
- `requirements-cpu.txt` is for CPU-only installs, and `requirements-cuda.txt` is for NVIDIA CUDA installs.
- A cached tokenizer model is written to `tokenizer_model.json` and automatically rebuilt when the training text changes.
- Checkpoints store the model state, optimizer state, loss history, and RNG state so a resumed run can continue from the same training trajectory.
