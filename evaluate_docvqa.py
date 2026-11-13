
import json
import csv
import re
from pathlib import Path
from peft import PeftModel
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ========== 1. 配置 ==========
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
VAL_FILE = Path("data/multitask/val.jsonl")
OUTPUT_DIR = Path("results/docvqa")

MAX_NEW_TOKENS = 64
MAX_IMAGE_SIZE = 768
DEVICE = "cuda:0"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with VAL_FILE.open("r", encoding="utf-8") as f:
    samples = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Validation samples: {len(samples)}")

# ========== 2. 加载模型 ==========
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(
    model,
    "checkpoints/qwen3_vl_lora_docvqa_300",
    is_trainable=False,
)

model.eval()

# ========== 3. 文本规范化 ==========
def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ========== 4. 编辑距离 ==========
def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):
        current = [i]

        for j, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1

            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))

        previous = current

    return previous[-1]


# ========== 5. Exact Match / ANLS ==========
def score_prediction(prediction, references):
    pred = normalize(prediction)

    refs = [
        normalize(ref)
        for ref in references
        if str(ref).strip()
    ]

    if not refs:
        return 0.0, 0.0

    em = float(pred in refs)

    scores = []

    for ref in refs:
        denominator = max(len(pred), len(ref), 1)

        normalized_distance = (
            edit_distance(pred, ref) / denominator
        )

        # DocVQA ANLS：编辑距离阈值 0.5
        score = (
            1 - normalized_distance
            if normalized_distance < 0.5
            else 0.0
        )

        scores.append(score)

    return em, max(scores)


# ========== 6. 模型推理 ==========
@torch.inference_mode()
def predict(sample):
    with Image.open(sample["image"]) as img:
        image = img.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        image = image.copy()

    # 同时用于后续 QLoRA 训练的提问模板
    question = sample["instruction"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        "Answer briefly using only the "
                        "information in the document."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    generated_ids = output_ids[
        :, inputs["input_ids"].shape[1]:
    ]

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return prediction.strip()


# ========== 7. 验证 ==========
results = []
total_em = 0.0
total_anls = 0.0

for sample in tqdm(samples, desc="DocVQA Baseline"):
    prediction = predict(sample)

    references = sample.get(
        "all_answers",
        [sample["answer"]],
    )

    em, anls = score_prediction(
        prediction,
        references,
    )

    total_em += em
    total_anls += anls

    results.append({
        "image": sample["image"],
        "question": sample["instruction"],
        "reference": sample["answer"],
        "prediction": prediction,
        "em": em,
        "anls": anls,
    })

# ========== 8. 统计并保存 ==========
n = len(results)

if n == 0:
    raise ValueError("Validation set is empty")

mean_em = total_em / n * 100
mean_anls = total_anls / n * 100

output_csv = OUTPUT_DIR / "finetuned_predictions.csv"

with output_csv.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\n===== DocVQA BASELINE =====")
print(f"Exact Match: {mean_em:.2f}%")
print(f"ANLS: {mean_anls:.2f}%")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

print("\n===== Examples =====")

for row in results[:5]:
    print("\nQuestion:", row["question"])
    print("Reference:", row["reference"])
    print("Prediction:", row["prediction"])
    print("ANLS:", round(row["anls"], 3))

print("\nSaved:", output_csv)
