
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
INPUT_CSV = Path("results/qlora_predictions_short.csv")
OUTPUT_CSV = Path("results/qlora_clipscore_short.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with INPUT_CSV.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


@torch.inference_mode()
def score_caption(image, caption):
    inputs = processor(
        images=[image],
        text=[caption],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(DEVICE)

    outputs = model(**inputs)

    image_features = F.normalize(
        outputs.image_embeds.float(), dim=-1
    )
    text_features = F.normalize(
        outputs.text_embeds.float(), dim=-1
    )

    similarity = (
        image_features * text_features
    ).sum(dim=-1).item()

    return 2.5 * max(0.0, similarity)


base_scores = []
fine_scores = []

for row in tqdm(rows):
    with Image.open(row["image"]) as img:
        image = img.convert("RGB").copy()

    base = score_caption(image, row["baseline"])
    fine = score_caption(image, row["finetuned"])

    row["baseline_clipscore"] = base
    row["finetuned_clipscore"] = fine
    row["clipscore_change"] = fine - base

    base_scores.append(base)
    fine_scores.append(fine)


with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

base_mean = sum(base_scores) / len(base_scores)
fine_mean = sum(fine_scores) / len(fine_scores)

print("\n===== CLIPScore Evaluation =====")
print(f"Baseline:   {base_mean:.4f}")
print(f"Fine-tuned: {fine_mean:.4f}")
print(f"Change:     {fine_mean-base_mean:+.4f}")
print(f"Saved: {OUTPUT_CSV}")
