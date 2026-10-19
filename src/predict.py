"""Predict the class of a single plantar thermogram.

Example:
    python src/predict.py --checkpoint outputs/best_model.pt --image data/diabetic/DM001_L.png
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from dataset import build_transforms
from model import build_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(checkpoint["model_name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transforms(checkpoint["image_size"])["eval"]
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)[0].cpu()

    prediction = class_names[int(probs.argmax())]
    print(f"Prediction: {prediction}")
    for name, prob in zip(class_names, probs.tolist()):
        print(f"  {name:12s} {prob:.3f}")


if __name__ == "__main__":
    main()
