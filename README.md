# Homemade LLM

A small transformer-based language model built for learning and experimentation.

This project is intentionally simple, slow, and readable so you can study how a tiny transformer is trained, how it predicts the next character, and how its internal tensors evolve during a forward pass.

## Purpose

This repository exists for study.

Instead of hiding the logic behind a large framework, it keeps the main pieces visible:
- tokenization
- batching
- self-attention
- training loss
- next-character probabilities
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

Shape walkthrough only:

```powershell
python tiny_llm_visual_debug.py --debug-only
```

## Study Flow

1. Edit `training_text.py` to choose a tiny training corpus.
2. Run `--debug-only` to inspect tensor shapes before training.
3. Run `--steps 2 --debug-shapes --skip-plots` for a very short training pass.
4. Run the full script when you want to watch learning and plots over time.

## Files

- `tiny_llm_visual_debug.py`: main model, training loop, and debug tools
- `training_text.py`: the text corpus used for training
- `STUDY_NOTES.md`: a place to write your own explanations while learning

## Notes

- The model is character-level, so it can only predict characters that already exist in the training text.
- The code is intentionally not optimized for performance. Readability comes first.
- `requirements-cpu.txt` is for CPU-only installs, and `requirements-cuda.txt` is for NVIDIA CUDA installs.
