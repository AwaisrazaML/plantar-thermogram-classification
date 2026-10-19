"""Turn training outputs into README-ready figures.

Reads outputs/metrics.json and outputs/history.json and writes:
    outputs/confusion_matrix.png
    outputs/training_curves.png

Example:
    python src/plot_results.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(metrics: dict, path: Path) -> None:
    matrix = np.array(metrics["confusion_matrix"])
    names = metrics["class_names"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colour = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=colour, fontsize=13)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test set confusion matrix ({metrics['model']})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(9, 3.5))
    loss_ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    loss_ax.plot(epochs, [h["val_loss"] for h in history], label="validation")
    loss_ax.set_title("Loss")
    f1_ax.plot(epochs, [h["train_f1"] for h in history], label="train")
    f1_ax.plot(epochs, [h["val_f1"] for h in history], label="validation")
    f1_ax.set_title("F1 score")
    f1_ax.set_ylim(0, 1.05)
    for ax in (loss_ax, f1_ax):
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()
    out = Path(args.output_dir)

    metrics = json.loads((out / "metrics.json").read_text())
    history = json.loads((out / "history.json").read_text())
    plot_confusion_matrix(metrics, out / "confusion_matrix.png")
    plot_training_curves(history, out / "training_curves.png")

    print("| Metric | Score |\n|---|---|")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        value = metrics.get(key)
        print(f"| {key} | {value:.3f} |" if value is not None else f"| {key} | n/a |")
    print(f"\nSaved {out / 'confusion_matrix.png'} and {out / 'training_curves.png'}")


if __name__ == "__main__":
    main()
