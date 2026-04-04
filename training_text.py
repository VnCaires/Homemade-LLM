from pathlib import Path


ROOT = Path(__file__).parent
TEXT_PATH = ROOT / "training_text.txt"

TRAINING_TEXT = TEXT_PATH.read_text(encoding="utf-8")
