
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# 1. 基本配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

BASELINE = "openai/clip-vit-base-patch32"
FINETUNED = "checkpoints/clip_finetune/best"

TEST_CSV = "data/test.csv"
IMAGE_DIR = Path("images")


# 2. 读取测试集
with open(TEST_CSV, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Test samples: {len(rows)}")


# 3. 定义模型评估函数
@torch.inference_mode()
def evaluate_model(model_path):

    print(f"\nEvaluating: {model_path}")

    model = CLIPModel.from_pretrained(model_path).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_path)

    model.eval()

    all_images = []
    all_texts = []

    # 逐批提取图片和文字特征
    for start in tqdm(range(0, len(rows), BATCH_SIZE)):

        batch = rows[start:start + BATCH_SIZE]

        images = []

        for row in batch:
            with Image.open(IMAGE_DIR / row["image"]) as img:
                images.append(img.convert("RGB").copy())

        texts = [row["caption"] for row in batch]

        inputs = processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(DEVICE)

        # 提取经过 Projection Head 的特征
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # 兼容部分 Transformers 版本返回的对象
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output

        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output

        all_images.append(
            F.normalize(image_features.float(), dim=-1).cpu()
        )

        all_texts.append(
            F.normalize(text_features.float(), dim=-1).cpu()
        )

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 拼接整个测试集的特征
    image_embeddings = torch.cat(all_images, dim=0)
    text_embeddings = torch.cat(all_texts, dim=0)

    # 计算所有图片和文本之间的相似度
    similarity = image_embeddings @ text_embeddings.T

    labels = torch.arange(len(rows))

    # 计算 Recall@K
    def recall_at_k(scores, k):
        topk = scores.topk(
            min(k, scores.shape[1]), dim=1
        ).indices

        correct = (topk == labels[:, None]).any(dim=1)

        return correct.float().mean().item()

    return {
        "I2T_R@1": recall_at_k(similarity, 1),
        "I2T_R@5": recall_at_k(similarity, 5),
        "T2I_R@1": recall_at_k(similarity.T, 1),
        "T2I_R@5": recall_at_k(similarity.T, 5)
    }


# 4. 分别测试两个模型
baseline_results = evaluate_model(BASELINE)
finetuned_results = evaluate_model(FINETUNED)


# 5. 输出对比结果
print("\n===== FINAL TEST RESULTS =====")

for metric in baseline_results:
    before = baseline_results[metric] * 100
    after = finetuned_results[metric] * 100

    print(
        f"{metric}: "
        f"Baseline={before:.2f}% | "
        f"Fine-tuned={after:.2f}% | "
        f"Change={after-before:+.2f} pp"
    )


# 6. 保存结果为 CSV
Path("results").mkdir(exist_ok=True)

with open("results/final_test.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Metric", "Baseline", "Fine-tuned", "Change_pp"
    ])

    for metric in baseline_results:
        before = baseline_results[metric] * 100
        after = finetuned_results[metric] * 100

        writer.writerow([
            metric, before, after, after - before
        ])

print("\nSaved: results/final_test.csv")
