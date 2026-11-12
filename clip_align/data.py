"""One-to-one image/caption dataset. CSV columns: image,caption."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class ImageTextDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer_name="distilbert-base-uncased", max_length=77):
        self.rows = pd.read_csv(csv_path)
        required = {"image", "caption"}
        if not required.issubset(self.rows.columns):
            raise ValueError("CSV must have image and caption columns")
        if self.rows.empty or self.rows[list(required)].isna().any().any():
            raise ValueError("CSV must contain nonempty image/caption pairs")
        self.image_dir = Path(image_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.image_dir / str(row["image"])
        with Image.open(image_path) as image:
            pixels = self.transform(image.convert("RGB"))
        tokens = self.tokenizer(
            str(row["caption"]), padding="max_length",
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        return {
            "image": pixels,  # [3, 224, 224]
            "input_ids": tokens["input_ids"].squeeze(0),  # [L]
            "attention_mask": tokens["attention_mask"].squeeze(0),  # [L]
        }
