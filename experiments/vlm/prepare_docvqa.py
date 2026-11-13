
import json
import random
from pathlib import Path

from datasets import load_dataset

# ========== 1. 配置 ==========
DATASET_NAME = "nielsr/docvqa_1200_examples"
OUTPUT_DIR = Path("data/multitask")
IMAGE_DIR = OUTPUT_DIR / "images"

TRAIN_COUNT = 300
VAL_COUNT = 50
SEED = 42

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ========== 2. 下载真实标注数据 ==========
dataset = load_dataset(DATASET_NAME)

print(dataset)
print("Columns:", dataset["train"].column_names)

# 只使用官方 train split
# 保留官方 test split，避免实验数据泄漏
train_data = dataset["train"]

indices = list(range(len(train_data)))
random.Random(SEED).shuffle(indices)

train_indices = indices[:TRAIN_COUNT]
val_indices = indices[
    TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
]

# ========== 3. 转换为统一 SFT 格式 ==========
def convert_split(name, selected_indices):
    output_path = OUTPUT_DIR / f"{name}.jsonl"

    with output_path.open(
        "w", encoding="utf-8"
    ) as f:
        for idx in selected_indices:
            sample = train_data[idx]

            # DocVQA 提供多个语言版本的问题
            question = sample["query"]["en"]

            # 使用人工标注答案，不使用 OCR 匹配产生的辅助字段
            answers = sample["answers"]
            if not question or not answers:
                continue

            answer = str(answers[0]).strip()
            if not answer:
                continue

            image_path = (
                IMAGE_DIR / f"{name}_{idx:05d}.jpg"
            ).resolve()

            sample["image"].convert("RGB").save(
                image_path,
                quality=95
            )

            record = {
                "image": str(image_path),
                "instruction": question,
                "answer": answer,
                "task": "docvqa",
                "source": DATASET_NAME,
                "source_id": str(sample["id"]),
                "all_answers": [
                    str(a) for a in answers
                ],
            }

            f.write(
                json.dumps(
                    record, ensure_ascii=False
                ) + "\n"
            )

    print(f"{name}: {len(selected_indices)} selected")
    print("Saved:", output_path)


convert_split("train", train_indices)
convert_split("val", val_indices)

print("\nDocVQA data preparation finished!")
