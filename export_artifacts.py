"""
Run this once after training to save deployment artifacts.
"""

import json
from pathlib import Path

import tensorflow as tf


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def save_artifacts(model: tf.keras.Model, class_names: list[str]) -> None:
    model.save(ARTIFACT_DIR / "ewaste_classifier.keras")
    with open(ARTIFACT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print("Saved artifacts in ./artifacts")


if __name__ == "__main__":
    raise SystemExit(
        "Import this file in your notebook and call save_artifacts(model, class_names)."
    )
