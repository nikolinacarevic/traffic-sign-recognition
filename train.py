from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import TrafficSignCNN


class MapTransform(Dataset):
    # Dodavanje transformacija na subset.

    def __init__(self, subset: Dataset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, target = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )


def load_signnames(project_dir: Path, class_folders: list[str]) -> list[str]:
    # Učitaj prikazna imena iz signnames.csv; redoslijed mora odgovarati folderima.
    path = project_dir / "signnames.csv"
    by_folder: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_folder[row["Folder"]] = row["SignName"]

    missing = [c for c in class_folders if c not in by_folder]
    if missing:
        raise RuntimeError(
            f"U signnames.csv nedostaju mape: {missing}. "
            f"Klase u datasetu: {class_folders}"
        )
    return [by_folder[c] for c in class_folders]


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    # Treniranje ili evaluacija modela na jednom epochu.
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    if train:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    else:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Treniranje CNN na vlastitom datasetu znakova")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Mapa s klasama (podmape = klase)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset mapa ne postoji: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    base = ImageFolder(root=str(data_dir), transform=None)
    class_folders = list(base.classes)
    num_classes = len(class_folders)
    if num_classes < 2:
        raise RuntimeError(f"Očekujem barem 2 klase u {data_dir}, našao {num_classes}")

    sign_names = load_signnames(project_dir, class_folders)
    print(f"Dataset: {data_dir}")
    print(f"Klasa: {num_classes}, slika: {len(base)}")
    for i, (folder, name) in enumerate(zip(class_folders, sign_names)):
        print(f"  {i}: {folder} -> {name}")

    n_total = len(base)
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        base, [n_train, n_val], generator=generator
    )

    train_ds = MapTransform(train_subset, train_transform())
    val_ds = MapTransform(val_subset, eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = TrafficSignCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = project_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"
    meta_path = ckpt_dir / "config.json"

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_classes": num_classes,
                        "img_size": 32,
                        "sign_names": sign_names,
                        "class_folders": class_folders,
                        "data_dir": str(data_dir.resolve()),
                        "best_val_accuracy": best_val_acc,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  -> spremljen najbolji model (val_acc={best_val_acc:.4f})")

    print(f"Gotovo. Najbolja val točnost: {best_val_acc:.4f}")
    print(f"Model: {best_path}")


if __name__ == "__main__":
    main()
