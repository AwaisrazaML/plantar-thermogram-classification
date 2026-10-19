"""Train and evaluate a plantar thermogram classifier.

Example:
    python src/train.py --data-dir data --model resnet18 --epochs 30 --subject-regex "^([A-Za-z]+\\d+)"
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ThermogramDataset, build_transforms, discover_images, split_by_subject, subject_ids
from model import SUPPORTED_MODELS, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--model", default="resnet18", choices=SUPPORTED_MODELS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patience", type=int, default=7, help="early stopping patience (epochs)")
    p.add_argument("--subject-regex", default=None, help="regex with one group extracting subject ID from file name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch instead of ImageNet weights")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the loader. Trains if an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_pred, y_prob = 0.0, [], [], []

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "eval"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(probs.argmax(1).cpu().tolist())
            y_prob.extend(probs.detach().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    average = "binary" if num_classes == 2 else "macro"
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        probs = np.array(y_prob)
        metrics["roc_auc"] = (
            roc_auc_score(y_true, probs[:, 1]) if num_classes == 2
            else roc_auc_score(y_true, probs, multi_class="ovr")
        )
    except ValueError:  # only one class present in y_true
        metrics["roc_auc"] = None
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = discover_images(args.data_dir)
    subjects = subject_ids(paths, labels, args.subject_regex)
    splits = split_by_subject(paths, labels, subjects, seed=args.seed)
    for name, (_, split_labels) in splits.items():
        counts = {class_names[k]: v for k, v in sorted(Counter(split_labels).items())}
        print(f"{name:5s}: {len(split_labels):4d} images {counts}")

    tfms = build_transforms(args.image_size)
    loaders = {
        name: DataLoader(
            ThermogramDataset(*splits[name], transform=tfms["train" if name == "train" else "eval"]),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
        )
        for name in ("train", "val", "test")
    }

    # class weights counter imbalance between control and diabetic groups
    counts = Counter(splits["train"][1])
    weights = torch.tensor(
        [len(splits["train"][1]) / (len(class_names) * counts.get(i, 1)) for i in range(len(class_names))],
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1, epochs_without_improvement, history = -1.0, 0, []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, *train_out = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, *val_out = run_epoch(model, loaders["val"], criterion, device)
        train_f1 = compute_metrics(*train_out, len(class_names))["f1"]
        val_metrics = compute_metrics(*val_out, len(class_names))
        scheduler.step(val_metrics["f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} f1 {train_f1:.3f} "
            f"| val loss {val_loss:.4f} f1 {val_metrics['f1']:.3f} acc {val_metrics['accuracy']:.3f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1, epochs_without_improvement = val_metrics["f1"], 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "class_names": class_names,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    # final evaluation on the held-out test set with the best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    _, *test_out = run_epoch(model, loaders["test"], criterion, device)
    test_metrics = compute_metrics(*test_out, len(class_names))
    test_metrics.update({"class_names": class_names, "model": args.model, "best_val_f1": best_f1})

    (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print("\nTest results:")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "class_names"}, indent=2))


if __name__ == "__main__":
    main()
