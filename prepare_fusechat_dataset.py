import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("fusechat_v1_clean_split_2048_filter_wrong.json")
DEFAULT_OUTPUT = Path("training_corpus.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flatten a local FuseChat JSON file into one plain-text training corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the downloaded FuseChat JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Plain-text file to write for training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="How many conversations to include in the output corpus.",
    )
    return parser.parse_args()


def flatten_row(row):
    parts = []
    for turn in row["conversations"]:
        role = turn["from"].strip().upper()
        text = turn["value"].strip()
        if text:
            parts.append(f"{role}: {text}")
    return "\n".join(parts)


def main():
    args = parse_args()

    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("Expected the JSON file to contain a top-level list.")

    subset = rows[: args.max_samples]
    conversations = []
    for row in subset:
        text = flatten_row(row)
        if text:
            conversations.append(text)

    if not conversations:
        raise ValueError("No non-empty conversations were found in the selected rows.")

    separator = "\n\n" + ("-" * 60) + "\n\n"
    corpus = separator.join(conversations)
    args.output.write_text(corpus, encoding="utf-8")

    print(f"Wrote {len(conversations)} conversations to {args.output}")
    print(f"Characters: {len(corpus)}")
    print(f"Approx. words: {len(corpus.split())}")
    print("Run your model normally after creating the corpus.")


if __name__ == "__main__":
    main()
