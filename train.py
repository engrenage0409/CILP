"""Train the projection heads of a minimal CLIP-style model."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clip_align import CLIPModel, ImageTextDataset, clip_loss
from clip_align.evaluation import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default="checkpoints/best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2 for contrastive learning")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set = ImageTextDataset(args.train_csv, args.image_dir, max_length=args.max_length)
    val_set = ImageTextDataset(args.val_csv, args.image_dir, max_length=args.max_length)
    if len(train_set) < 2:
        raise ValueError("Training set requires at least two image-text pairs")
    if len(val_set) < 2:
        raise ValueError("Validation set requires at least two image-text pairs")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = CLIPModel(embed_dim=args.embed_dim).to(device)
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr, weight_decay=0.01,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen ResNet batchnorm statistics and text dropout unchanged.
        model.image_encoder.eval()
        model.text_encoder.eval()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            if batch["image"].shape[0] < 2:
                continue  # Last singleton batch has no in-batch negatives.
            images = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, ids, masks)
            loss = clip_loss(logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        if steps == 0:
            raise ValueError("No train batches with >=2 samples")
        metrics = evaluate_retrieval(model, val_loader, device)
        score = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2
        print(f"Epoch {epoch}/{args.epochs} loss={total_loss / steps:.4f} "
              + " ".join(f"{k}={v:.2%}" for k, v in metrics.items()))
        if score > best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "embed_dim": args.embed_dim,
                "max_length": args.max_length,
                "epoch": epoch,
                "metrics": metrics,
            }, output)
            print(f"Saved best checkpoint: {output}")


if __name__ == "__main__":
    main()
