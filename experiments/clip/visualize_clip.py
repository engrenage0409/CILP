
import csv
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ========== 1. 配置 ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MODEL_NAME = "openai/clip-vit-base-patch32"
FINETUNED = "checkpoints/clip_finetune/best"

RESULT_DIR = Path("results/visualization")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

with open("data/test.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} test pairs")


# ========== 2. 训练历史 ==========
# 以下数据来自你实际运行的 5 Epoch 实验
history = {
    "epoch": [1, 2, 3, 4, 5],
    "loss": [0.1308, 0.1202, 0.1110, 0.1066, 0.0963],
    "I2T_R1": [59.2, 59.4, 59.6, 60.5, 60.7],
    "I2T_R5": [86.4, 87.2, 87.8, 87.9, 88.1],
    "T2I_R1": [57.4, 57.5, 58.2, 58.8, 59.4],
    "T2I_R5": [83.1, 84.0, 84.5, 84.2, 84.4],
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history["epoch"], history["loss"], marker="o")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("InfoNCE Loss")
axes[0].set_xticks(history["epoch"])
axes[0].grid(alpha=0.3)

for key in ["I2T_R1", "I2T_R5", "T2I_R1", "T2I_R5"]:
    axes[1].plot(
        history["epoch"], history[key],
        marker="o", label=key
    )

axes[1].set_title("Validation Recall@K")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Recall (%)")
axes[1].set_xticks(history["epoch"])
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULT_DIR / "training_curves.png", dpi=200)
plt.close(fig)


# ========== 3. 独立测试集指标对比 ==========
metrics = ["I2T R@1", "I2T R@5", "T2I R@1", "T2I R@5"]

baseline_values = [57.3, 83.4, 54.6, 81.4]
finetuned_values = [59.9, 86.2, 58.1, 85.2]

import numpy as np

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(x - width / 2, baseline_values, width, label="Original CLIP")
ax.bar(x + width / 2, finetuned_values, width, label="Fine-tuned CLIP")

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Recall (%)")
ax.set_title("Flickr8k Test Set: Baseline vs Fine-tuned")
ax.set_ylim(0, 100)
ax.legend()

for i in range(len(metrics)):
    ax.text(i - width / 2, baseline_values[i] + 1,
            f"{baseline_values[i]:.1f}", ha="center")
    ax.text(i + width / 2, finetuned_values[i] + 1,
            f"{finetuned_values[i]:.1f}", ha="center")

plt.tight_layout()
plt.savefig(RESULT_DIR / "baseline_vs_finetuned.png", dpi=200)
plt.close(fig)


# ========== 4. 模型推理 ==========
@torch.inference_mode()
def get_similarity(model_path):
    model = CLIPModel.from_pretrained(model_path).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()

    image_vectors = []
    text_vectors = []

    for start in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[start:start + BATCH_SIZE]

        images = []
        for row in batch:
            with Image.open(Path("images") / row["image"]) as img:
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

        output = model(**inputs)

        image_vectors.append(
            F.normalize(output.image_embeds.float(), dim=-1).cpu()
        )
        text_vectors.append(
            F.normalize(output.text_embeds.float(), dim=-1).cpu()
        )

    del model

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    image_vectors = torch.cat(image_vectors)
    text_vectors = torch.cat(text_vectors)

    return image_vectors @ text_vectors.T


print("Evaluating original CLIP...")
baseline_sim = get_similarity(MODEL_NAME)

print("Evaluating fine-tuned CLIP...")
finetuned_sim = get_similarity(FINETUNED)



# ========== 5. Similarity Heatmap ==========

n = min(12, len(rows))

# 取出前 12 对图文的相似度矩阵
baseline_matrix = baseline_sim[:n, :n].numpy()
finetuned_matrix = finetuned_sim[:n, :n].numpy()

# 两个矩阵的统一颜色范围
vmin = min(baseline_matrix.min(), finetuned_matrix.min())
vmax = max(baseline_matrix.max(), finetuned_matrix.max())

# 计算微调带来的相似度变化
difference = finetuned_matrix - baseline_matrix

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 原始 CLIP
sns.heatmap(
    baseline_matrix,
    ax=axes[0],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    square=True
)
axes[0].set_title("Original CLIP")

# 微调后 CLIP
sns.heatmap(
    finetuned_matrix,
    ax=axes[1],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    square=True
)
axes[1].set_title("Fine-tuned CLIP")

# 相似度变化
limit = max(abs(difference.min()), abs(difference.max()))

sns.heatmap(
    difference,
    ax=axes[2],
    cmap="coolwarm",
    center=0,
    vmin=-limit,
    vmax=limit,
    square=True
)
axes[2].set_title("Similarity Difference (Fine-tuned - Original)")

for ax in axes:
    ax.set_xlabel("Text Index")
    ax.set_ylabel("Image Index")

plt.tight_layout()
plt.savefig(
    RESULT_DIR / "similarity_heatmap.png",
    dpi=200
)
plt.close(fig)



# ========== 6. Image-to-Text Top-5 ==========
num_examples = min(3, len(rows))
fig, axes = plt.subplots(num_examples, 2, figsize=(13, 4 * num_examples))
axes = np.atleast_2d(axes)

for i in range(num_examples):
    with Image.open(Path("images") / rows[i]["image"]) as img:
        axes[i, 0].imshow(img.convert("RGB"))

    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Image Query #{i}")

    top5 = finetuned_sim[i].topk(
        min(5, len(rows))
    ).indices.tolist()

    lines = []

    for rank, idx in enumerate(top5, 1):
        mark = "[GT]" if idx == i else ""
        caption = textwrap.fill(rows[idx]["caption"], width=60)
        score = finetuned_sim[i, idx].item()

        lines.append(
            f"{rank}. {caption}\n"
            f"   Similarity: {score:.3f} {mark}"
        )

    axes[i, 1].text(
        0, 0.98,
        "\n\n".join(lines),
        va="top",
        fontsize=10,
        transform=axes[i, 1].transAxes
    )
    axes[i, 1].axis("off")
    axes[i, 1].set_title("Top-5 Retrieved Captions")

plt.tight_layout()
plt.savefig(RESULT_DIR / "retrieval_i2t.png", dpi=200)
plt.close(fig)


# ========== 7. Text-to-Image Top-5 ==========

num_examples = min(3, len(rows))
top_k = min(5, len(rows))

fig, axes = plt.subplots(
    num_examples,
    top_k,
    figsize=(16, 4.5 * num_examples),
    squeeze=False
)

for i in range(num_examples):

    # 当前文本 Query
    query = rows[i]["caption"]

    # 获取 Top-5 图像索引
    top5 = finetuned_sim[:, i].topk(
        top_k
    ).indices.tolist()

    for rank, idx in enumerate(top5):
        ax = axes[i, rank]

        # 打开检索图片
        image_path = Path("images") / rows[idx]["image"]

        with Image.open(image_path) as img:
            ax.imshow(img.convert("RGB"))

        ax.axis("off")

        # 读取相似度
        score = finetuned_sim[idx, i].item()

        # 标记正确图片
        is_correct = (idx == i)

        ax.set_title(
            f"Rank {rank + 1} | {score:.3f}"
            + (" | GT" if is_correct else ""),
            fontsize=10,
            color="green" if is_correct else "black"
        )

    # 在当前行上方显示 Query
    axes[i, 0].text(
        0,
        1.22,
        f"Query {i}: {query}",
        transform=axes[i, 0].transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        clip_on=False
    )

plt.subplots_adjust(
    hspace=0.75,
    wspace=0.15,
    top=0.93
)

plt.savefig(
    RESULT_DIR / "retrieval_t2i.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print(f"\nVisualization finished: {RESULT_DIR}")