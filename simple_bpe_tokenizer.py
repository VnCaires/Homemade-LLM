import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PIECE_PATTERN = re.compile(r"\s+\w+|\w+|\s+[^\w\s]+|[^\w\s]+|\s+", re.UNICODE)


def split_text_into_pieces(text: str):
    return PIECE_PATTERN.findall(text)


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
