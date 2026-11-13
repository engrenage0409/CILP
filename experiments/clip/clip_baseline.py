
import csv                                # 读取 CSV 数据
import torch                              # PyTorch
import torch.nn.functional as F           # 特征归一化
from pathlib import Path                  # 文件路径处理
from PIL import Image                     # 图片读取
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm                     # 显示进度条


# 1. 基础配置
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4                            # 适合先在 6GB 可用显存上测试

print("Using:", DEVICE)


# 2. 下载官方 CLIP 模型及预处理器
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()                               # 评估模式，不训练权重


# 3. 读取与之前相同的验证集
with open("data/val.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

image_embeddings = []                      # 存储图片特征
text_embeddings = []                       # 存储文字特征


# 4. 分 Batch 提取特征
with torch.inference_mode():                # 禁止梯度，节约显存

    for start in tqdm(range(0, len(rows), BATCH_SIZE)):

        batch = rows[start:start + BATCH_SIZE]  # 取出一个 Batch

        images = []                        # 当前 Batch 的图片

        for row in batch:
            path = Path("images") / row["image"]

            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())

        texts = [row["caption"] for row in batch]  # 读取对应描述

        inputs = processor(
            text=texts,                     # 文本 Tokenization
            images=images,                   # 图像 Resize / Normalize
            padding=True,                    # 对齐文本长度
            truncation=True,                 # 截断超长文本
            max_length=77,                   # CLIP 最大文本长度
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**inputs)            # 执行官方 CLIP Forward

        img_emb = F.normalize(
            outputs.image_embeds, dim=-1     # [B,512] 图像特征
        )

        txt_emb = F.normalize(
            outputs.text_embeds, dim=-1      # [B,512] 文本特征
        )

        image_embeddings.append(img_emb.cpu()) # 保存到 CPU
        text_embeddings.append(txt_emb.cpu())


# 5. 拼接整个验证集的特征
image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)

# [N,512] × [512,N] → [N,N]
similarity = image_embeddings @ text_embeddings.T


# 6. 定义 Recall@K
def recall_at_k(scores, k):
    k = min(k, scores.shape[1])              # 防止 K 超过候选数量

    labels = torch.arange(scores.shape[0])   # 正确配对为对角线

    topk = scores.topk(k, dim=1).indices     # 取前 K 个下标

    correct = (topk == labels[:, None])      # 是否找到正确下标

    return correct.any(dim=1).float().mean().item()


# 7. 分别计算图找文、文找图
results = {
    "I2T_R@1": recall_at_k(similarity, 1),
    "I2T_R@5": recall_at_k(similarity, 5),
    "T2I_R@1": recall_at_k(similarity.T, 1),
    "T2I_R@5": recall_at_k(similarity.T, 5)
}

print("\nOfficial CLIP Zero-shot Baseline")

for name, score in results.items():
    print(f"{name}: {score * 100:.2f}%")


# 8. 保存评估结果
Path("results").mkdir(exist_ok=True)

with open("results/clip_baseline.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])

    for name, score in results.items():
        writer.writerow([name, score])
