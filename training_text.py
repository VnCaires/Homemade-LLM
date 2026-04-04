from pathlib import Path


CORPUS_PATH = Path(__file__).with_name("training_corpus.txt")

TRAINING_TEXT = CORPUS_PATH.read_text(encoding="utf-8")
