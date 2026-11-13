
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


# ========= 1. 配置 =========
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-6

assert DEVICE == "cuda", "请先检查 PyTorch CUDA 环境"


# ========= 2. 加载预训练 CLIP =========
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# 先冻结全部参数
for param in model.parameters():
    param.requires_grad = False

# 只解冻 Text Encoder 最后一个 Transformer Block
#for param in model.text_model.encoder.layers[-1].parameters():
 #   param.requires_grad = True

# 解冻两个 Projection Heads
for param in model.visual_projection.parameters():
    param.requires_grad = True

for param in model.text_projection.parameters():
    param.requires_grad = True

# 解冻相似度温度参数
model.logit_scale.requires_grad = True

print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


# ========= 3. Dataset =========
class FlickrDataset(Dataset):
    def __init__(self, csv_file):
        with open(csv_file, encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        with Image.open(Path("images") / row["image"]) as img:
            image = img.convert("RGB").copy()

        return image, row["caption"]


# ========= 4. DataLoader =========
def collate_fn(batch):
    images, texts = zip(*batch)

    return processor(
        images=list(images),
        text=list(texts),
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )


train_loader = DataLoader(
    FlickrDataset("data/train.csv"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    FlickrDataset("data/val.csv"),
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ========= 5. Optimizer =========
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
    weight_decay=0.01
)


# ========= 6. InfoNCE Loss =========
def contrastive_loss(logits):
    labels = torch.arange(
        logits.shape[0], device=logits.device
    )

    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2


# ========= 7. Validation =========
@torch.no_grad()
def validate():
    model.eval()

    all_images = []
    all_texts = []

    for batch in val_loader:
        batch = batch.to(DEVICE)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)

        all_images.append(
            F.normalize(output.image_embeds.float(), dim=-1).cpu()
        )
        all_texts.append(
            F.normalize(output.text_embeds.float(), dim=-1).cpu()
        )

    image_emb = torch.cat(all_images)
    text_emb = torch.cat(all_texts)

    sim = image_emb @ text_emb.T
    labels = torch.arange(len(image_emb))

    def recall(scores, k):
        topk = scores.topk(min(k, scores.shape[1]), dim=1).indices
        return (topk == labels[:, None]).any(dim=1).float().mean()

    return {
        "I2T_R@1": recall(sim, 1).item(),
        "I2T_R@5": recall(sim, 5).item(),
        "T2I_R@1": recall(sim.T, 1).item(),
        "T2I_R@5": recall(sim.T, 5).item()
    }


# ========= 8. Training Loop =========
Path("checkpoints/clip_finetune").mkdir(
    parents=True, exist_ok=True
)

best_score = -1

for epoch in range(EPOCHS):
    model.train()

    # 冻结的视觉网络保持 eval 模式
    model.vision_model.eval()

    total_loss = 0
    steps = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        if batch["input_ids"].shape[0] < 2:
            continue

        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        # BF16 混合精度前向传播
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
            loss = contrastive_loss(output.logits_per_image)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    metrics = validate()

    avg_loss = total_loss / steps
    score = (metrics["I2T_R@1"] + metrics["T2I_R@1"]) / 2

    print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f}")

    for name, value in metrics.items():
        print(f"{name}: {value:.2%}")

    if score > best_score:
        best_score = score

        model.save_pretrained("checkpoints/clip_finetune/best")
        processor.save_pretrained("checkpoints/clip_finetune/best")

        print("Saved best fine-tuned CLIP!")

print("Training finished!")
