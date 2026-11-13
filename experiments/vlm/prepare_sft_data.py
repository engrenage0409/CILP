
import csv
import json
from pathlib import Path

# 之前的 CLIP 项目路径
CLIP_DIR = Path.cwd()  # Run with: python -m experiments.vlm.prepare_sft_data from repo root

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# 先做小规模实验，确认训练能跑通
SPLITS = {
    "train": 200,
    "val": 50,
}

import random

random.seed(42)

INSTRUCTIONS = [
    "Describe this image.",
    "What is happening in this picture?",
    "Write a short caption for this photo.",
    "Describe the main objects and actions in the image.",
    "What can you see in this photograph?",
    "Summarize the visual content of this image.",
    "Provide a concise description of the scene.",
    "Describe the scene shown in this picture."
]

for split, limit in SPLITS.items():
    csv_path = CLIP_DIR / "data" / f"{split}.csv"
    output_path = OUTPUT_DIR / f"{split}.jsonl"

    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    samples = []

    for row in rows[:limit]:
        image_path = (
            CLIP_DIR / "images" / row["image"]
        ).resolve()

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # 一张图片 + 一条问题 + 一条真实 Caption
        samples.append({
            "image": str(image_path),
            "instruction": random.choice(INSTRUCTIONS),
            "answer": row["caption"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(
                json.dumps(sample, ensure_ascii=False)
                + "\n"
            )

    print(f"{split}: {len(samples)} samples -> {output_path}")
