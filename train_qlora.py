
import json
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ========== 1. 配置 ==========
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
TRAIN_FILE = "data/multitask/train.jsonl"
OUTPUT_DIR = Path("checkpoints/qwen3_vl_lora_docvqa_300")

SEED = 42
SMOKE_TEST = False
MAX_SAMPLES = 10 if SMOKE_TEST else 300

EPOCHS = 1
GRAD_ACCUM = 8
LR = 1e-4

torch.manual_seed(SEED)
random.seed(SEED)

assert torch.cuda.is_available(), "需要 CUDA GPU"
DEVICE = torch.device("cuda:0")

# ========== 2. 加载训练数据 ==========
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

samples = samples[:MAX_SAMPLES]
print(f"Training samples: {len(samples)}")

# ========== 3. 加载 4-bit Qwen3-VL ==========
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.config.use_cache = False

# ========== 4. 准备 QLoRA ==========
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# 仅给语言模型的 Q/V 投影添加 LoRA
target_names = [
    name
    for name, module in model.named_modules()
    if "language_model" in name
    and ".layers." in name
    and name.endswith((".q_proj", ".v_proj"))
]

if not target_names:
    raise RuntimeError("未找到语言模型 Q/V 投影层")

print(f"LoRA target modules: {len(target_names)}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_names,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.train()

# ========== 5. 优化器 ==========
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
)

# ========== 6. 构造一条多模态训练样本 ==========
def prepare_sample(sample):
    # 读取并缩小图片，优先保证8GB显存能跑通
    with Image.open(sample["image"]) as img:
        image = img.convert("RGB")
        image.thumbnail((768,768))
        image = image.copy()

    # 用户消息：图片 + 指令
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
               {
                "type": "text",
                "text": (
                    f"{sample['instruction']}\n"
                    "Answer briefly using only the "
                    "information in the document."
                ),
                },
            ],
        }
    ]

    # 完整对话：用户消息 + Assistant 答案
    full_messages = user_messages + [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["answer"]}
            ],
        }
    ]

    # Prompt 包含 assistant 开始生成的标记
    prompt_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 完整训练文本包含标准答案
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 两次处理使用完全相同的图片
    prompt_inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )

    inputs = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt",
    )

    prompt_ids = prompt_inputs["input_ids"][0]
    full_ids = inputs["input_ids"][0]

    # 检查两次编码后的 Prompt 前缀是否一致
    prompt_len = prompt_ids.shape[0]

    if not torch.equal(full_ids[:prompt_len], prompt_ids):
        raise RuntimeError(
            "Prompt 与完整对话的 Token 前缀不一致，"
            "请检查 chat template 和 processor。"
        )

    # 标签复制自完整对话
    labels = inputs["input_ids"].clone()

    # 用户问题、图片 Tokens、Assistant Header 均不计入 Loss
    labels[:, :prompt_len] = -100

    # 防止没有任何有效答案 Token
    if (labels != -100).sum().item() == 0:
        raise RuntimeError("No assistant tokens available for training")

    inputs["labels"] = labels

    # 输入移动到 GPU
    return inputs.to(DEVICE)


# ========== 7. 训练 ==========
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):
    random.shuffle(samples)
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    updates = 0

    progress = tqdm(samples, desc=f"Epoch {epoch+1}")

    for step, sample in enumerate(progress):
        batch = prepare_sample(sample)

        # Forward：计算标准答案的 Cross-Entropy Loss
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step+1}")

        running_loss += loss.item()

        # Backward：只对 LoRA A/B 求梯度
        window_start = (step // GRAD_ACCUM) * GRAD_ACCUM
        window_size = min(GRAD_ACCUM, len(samples) - window_start)

        (loss / window_size).backward()

        # 每累计 GRAD_ACCUM 条样本更新一次
        if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(samples):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

        del batch, outputs, loss

    print(
        f"Epoch {epoch+1} | "
        f"Mean Loss: {running_loss / len(samples):.4f} | "
        f"Optimizer Updates: {updates}"
    )

    if torch.cuda.is_available():
        print(
            "Peak allocated VRAM (GB):",
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        )

# ========== 8. 保存 LoRA Adapter ==========
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nLoRA Adapter saved to: {OUTPUT_DIR}")
print("Training finished!")
