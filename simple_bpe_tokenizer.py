import hashlib
import json
import re
import argparse
from collections import Counter
from pathlib import Path


PIECE_PATTERN = re.compile(r"\s+\w+|\w+|\s+[^\w\s]+|[^\w\s]+|\s+", re.UNICODE)


def split_text_into_pieces(text: str):
    return PIECE_PATTERN.findall(text)


def text_to_display(text_value: str):
    return text_value.encode("unicode_escape").decode("ascii")


def merge_token_sequence(token_sequence, pair, new_token_id):
    merged = []
    i = 0
    while i < len(token_sequence):
        if i < len(token_sequence) - 1 and token_sequence[i] == pair[0] and token_sequence[i + 1] == pair[1]:
            merged.append(new_token_id)
            i += 2
        else:
            merged.append(token_sequence[i])
            i += 1
    return tuple(merged)


class SimpleBPETokenizer:
    def __init__(self, base_bytes, merges, model_path=None):
        self.base_bytes = list(base_bytes)
        self.base_vocab_size = len(self.base_bytes)
        self.byte_to_token_id = {byte: token_id for token_id, byte in enumerate(self.base_bytes)}
        self.merges = [tuple(pair) for pair in merges]
        self.model_path = Path(model_path) if model_path is not None else None
        self.token_bytes = {
            token_id: bytes([byte_value]) for token_id, byte_value in enumerate(self.base_bytes)
        }

        for new_token_id, pair in enumerate(self.merges, start=self.base_vocab_size):
            self.token_bytes[new_token_id] = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]

        self.vocab_size = len(self.token_bytes)
        self.num_merges = len(self.merges)
        self._piece_cache = {}

    @classmethod
    def train_or_load(cls, text: str, target_vocab_size: int, min_pair_frequency: int, model_path):
        model_path = Path(model_path)
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if model_path.exists():
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            if (
                payload.get("model_version") == 2
                and
                payload.get("text_hash") == expected_hash
                and payload.get("target_vocab_size") == target_vocab_size
                and payload.get("min_pair_frequency") == min_pair_frequency
            ):
                print(f"Loaded tokenizer from {model_path.name}")
                return cls(payload["base_bytes"], payload["merges"], model_path=model_path)

        print("Training byte-level BPE tokenizer...")
        tokenizer = cls.train_from_text(text, target_vocab_size=target_vocab_size, min_pair_frequency=min_pair_frequency)
        payload = {
            "model_version": 2,
            "text_hash": expected_hash,
            "target_vocab_size": target_vocab_size,
            "min_pair_frequency": min_pair_frequency,
            "base_bytes": tokenizer.base_bytes,
            "merges": [list(pair) for pair in tokenizer.merges],
        }
        model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tokenizer.model_path = model_path
        print(f"Saved tokenizer to {model_path.name}")
        return tokenizer

    @classmethod
    def train_from_text(cls, text: str, target_vocab_size: int, min_pair_frequency: int):
        pieces = split_text_into_pieces(text)
        base_bytes = sorted({byte for piece in pieces for byte in piece.encode("utf-8")})
        if target_vocab_size < len(base_bytes):
            raise ValueError("target_vocab_size must be at least the number of distinct bytes in the text.")

        byte_to_token_id = {byte_value: token_id for token_id, byte_value in enumerate(base_bytes)}
        piece_counter = Counter()
        for piece in pieces:
            piece_counter[tuple(byte_to_token_id[byte] for byte in piece.encode("utf-8"))] += 1

        merges = []
        vocab = piece_counter

        while len(base_bytes) + len(merges) < target_vocab_size:
            pair_counts = Counter()
            for token_sequence, count in vocab.items():
                if len(token_sequence) < 2:
                    continue
                for pair in zip(token_sequence, token_sequence[1:]):
                    pair_counts[pair] += count

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_frequency:
                break

            new_token_id = len(base_bytes) + len(merges)
            merges.append(best_pair)

            new_vocab = Counter()
            for token_sequence, count in vocab.items():
                merged_sequence = merge_token_sequence(token_sequence, best_pair, new_token_id)
                new_vocab[merged_sequence] += count
            vocab = new_vocab

        return cls(base_bytes, merges)

    def encode(self, text: str):
        unknown_chars = sorted(
            {char for char in text if any(byte not in self.byte_to_token_id for byte in char.encode("utf-8"))}
        )
        if unknown_chars:
            shown = ", ".join(repr(char) for char in unknown_chars[:10])
            raise ValueError(f"Prompt contains characters outside the tokenizer vocabulary: {shown}")

        token_ids = []
        for piece in split_text_into_pieces(text):
            if piece not in self._piece_cache:
                piece_tokens = tuple(self.byte_to_token_id[byte] for byte in piece.encode("utf-8"))
                for new_token_id, pair in enumerate(self.merges, start=self.base_vocab_size):
                    piece_tokens = merge_token_sequence(piece_tokens, pair, new_token_id)
                self._piece_cache[piece] = list(piece_tokens)
            token_ids.extend(self._piece_cache[piece])
        return token_ids

    def decode(self, token_ids):
        byte_string = b"".join(self.token_bytes[token_id] for token_id in token_ids)
        return byte_string.decode("utf-8", errors="replace")

    def token_to_text(self, token_id: int):
        return self.token_bytes[token_id].decode("utf-8", errors="replace")

    def token_to_display(self, token_id: int):
        return self.token_to_text(token_id).encode("unicode_escape").decode("ascii")

    def token_to_short_display(self, token_id: int, max_length: int = 16):
        token_text = self.token_to_display(token_id)
        if len(token_text) <= max_length:
            return token_text
        return token_text[: max_length - 3] + "..."

    def format_tokenization_report(
        self,
        text: str,
        max_pieces: int = 12,
        max_tokens_per_piece: int = 8,
        max_prompt_tokens: int = 32,
        token_display_length: int = 16,
    ):
        prompt_ids = self.encode(text)
        pieces = split_text_into_pieces(text)
        if not pieces:
            return ["Prompt tokenization: []"]

        lines = ["Prompt tokenization:"]
        for piece_index, piece in enumerate(pieces[:max_pieces], start=1):
            piece_ids = self.encode(piece)
            shown_piece_ids = piece_ids[:max_tokens_per_piece]
            shown_tokens = [
                self.token_to_short_display(token_id, max_length=token_display_length)
                for token_id in shown_piece_ids
            ]
            piece_id_suffix = " ..." if len(piece_ids) > max_tokens_per_piece else ""
            token_suffix = ["..."] if len(piece_ids) > max_tokens_per_piece else []
            lines.append(
                f" {piece_index:2d}. {repr(text_to_display(piece)):<18}"
                f" -> ids {shown_piece_ids}{piece_id_suffix}"
                f" -> tokens {shown_tokens + token_suffix}"
            )

        if len(pieces) > max_pieces:
            lines.append(f" ... {len(pieces) - max_pieces} more text pieces omitted")

        shown_prompt_ids = prompt_ids[:max_prompt_tokens]
        prompt_id_suffix = " ..." if len(prompt_ids) > max_prompt_tokens else ""
        lines.append(f"Full prompt token ids: {shown_prompt_ids}{prompt_id_suffix}")
        return lines

    def print_tokenization_report(
        self,
        text: str,
        max_pieces: int = 12,
        max_tokens_per_piece: int = 8,
        max_prompt_tokens: int = 32,
        token_display_length: int = 16,
    ):
        for line in self.format_tokenization_report(
            text,
            max_pieces=max_pieces,
            max_tokens_per_piece=max_tokens_per_piece,
            max_prompt_tokens=max_prompt_tokens,
            token_display_length=token_display_length,
        ):
            print(line)

    def format_vocabulary_report(self):
        lines = [
            f"# Vocabulary size: {self.vocab_size}",
            f"# Base tokens: {self.base_vocab_size}",
            f"# Merged tokens: {self.num_merges}",
            "",
        ]
        for token_id in range(self.vocab_size):
            token_kind = "base" if token_id < self.base_vocab_size else "merged"
            token_text = self.token_to_display(token_id)
            lines.append(f"{token_id}\t{token_kind}\t{repr(token_text)}")
        return lines

    def export_vocabulary(self, output_path: Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(self.format_vocabulary_report()) + "\n", encoding="utf-8")
        print(f"Exported vocabulary to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or load the local byte-level BPE tokenizer and show how a prompt becomes tokens."
    )
    parser.add_argument(
        "--text-path",
        type=Path,
        default=Path(__file__).with_name("training_text.txt"),
        help="Corpus used to train or load the tokenizer.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).with_name("tokenizer_model.json"),
        help="Where to save or load the tokenizer model.",
    )
    parser.add_argument(
        "--prompt",
        default="Call me",
        help="Prompt to tokenize for the study output.",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=1024,
        help="Target tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=3,
        help="Minimum pair frequency required to create a merge.",
    )
    parser.add_argument(
        "--export-vocab-path",
        type=Path,
        default=None,
        help="Optional path to save every token in the vocabulary to a text file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_text = args.text_path.read_text(encoding="utf-8")
    tokenizer = SimpleBPETokenizer.train_or_load(
        corpus_text,
        target_vocab_size=args.target_vocab_size,
        min_pair_frequency=args.min_pair_frequency,
        model_path=args.model_path,
    )

    print(f"Corpus path: {args.text_path}")
    print(f"Model path: {args.model_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Tokenizer merges: {tokenizer.num_merges}")
    print(f"Prompt: {repr(args.prompt)}")
    tokenizer.print_tokenization_report(args.prompt)
    if args.export_vocab_path is not None:
        tokenizer.export_vocabulary(args.export_vocab_path)


if __name__ == "__main__":
    main()
