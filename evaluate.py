"""Evaluate a saved model on held-out image/caption pairs."""
import argparse

import torch
from torch.utils.data import DataLoader

from clip_align import CLIPModel, ImageTextDataset
from clip_align.evaluation import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = ImageTextDataset(
        args.val_csv, args.image_dir, max_length=checkpoint["max_length"]
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = CLIPModel(embed_dim=checkpoint["embed_dim"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    metrics = evaluate_retrieval(model, loader, device)
    for name, value in metrics.items():
        print(f"{name}: {value:.2%}")


if __name__ == "__main__":
    main()
