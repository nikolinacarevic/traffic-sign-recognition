from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import TrafficSignCNN


def load_config(checkpoint_dir: Path) -> dict:
    meta = checkpoint_dir / "config.json"
    if not meta.is_file():
        raise FileNotFoundError(
            f"Nedostaje {meta}. Prvo pokreni train.py da generira config.json uz model."
        )
    with open(meta, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Predikcija prometnog znaka na slici")
    parser.add_argument("--image", "-i", type=str, required=True, help="Putanja do slike")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Putanja do best_model.pt (default: checkpoints/best_model.pt pokraj skripte)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Broj najvjerojatnijih klasa")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else project_dir / "checkpoints" / "best_model.pt"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Nema checkpointa: {ckpt_path}\nPokreni prvo: python train.py"
        )

    cfg = load_config(ckpt_path.parent)
    num_classes = cfg["num_classes"]
    img_size = cfg.get("img_size", 32)
    sign_names = cfg["sign_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Slika ne postoji: {img_path}")

    pil = Image.open(img_path).convert("RGB")
    x = tf(pil).unsqueeze(0).to(device)

    model = TrafficSignCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_k = min(args.top_k, num_classes)
    top_prob, top_idx = probs.topk(top_k)

    print(f"Slika: {img_path.resolve()}")
    print(f"Top-{top_k} predikcije:")
    for rank in range(top_k):
        idx = top_idx[rank].item()
        p = top_prob[rank].item()
        name = sign_names[idx]
        print(f"  {rank + 1}. Klasa {idx}: {name}  (vjerojatnost {p * 100:.2f}%)")


if __name__ == "__main__":
    main()
