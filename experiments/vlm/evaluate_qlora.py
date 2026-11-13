
import csv
import json
import gc
from pathlib import Path

import torch
import sacrebleu
from rouge_score import rouge_scorer
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ========== 1. 配置 ==========
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "checkpoints/qwen3_vl_lora_200"
VAL_FILE = "data/val.jsonl"
OUTPUT_DIR = Path("results")

MAX_SAMPLES = 50
MAX_NEW_TOKENS = 64
DEVICE = "cuda:0"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "CUDA GPU required"

with open(VAL_FILE, "r", encoding="utf-8") as f:
    samples = [
        json.loads(line)
        for line in f
        if line.strip()
    ][:MAX_SAMPLES]

print(f"Validation samples: {len(samples)}")


# ========== 2. 4-bit 量化配置 ==========
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)


# ========== 3. 加载模型 ==========
def load_model(use_lora=False):
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )

    if use_lora:
        print("Loading LoRA Adapter...")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            is_trainable=False,
        )

    model.config.use_cache = True
    model.eval()

    return model


# ========== 4. 单张图片推理 ==========
@torch.inference_mode()
def predict(model, sample):

    # 与训练时保持相同的图像尺寸处理
    with Image.open(sample["image"]) as img:
        image = img.convert("RGB")
        image.thumbnail((224, 224))
        image = image.copy()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                    "Write one concise sentence describing the image. "
                    "Mention the main subjects, their actions, and important "
                    "visual details. Use approximately 10 to 15 words."
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
        use_cache=True,
    )

    # 只解码模型新生成的答案
    answer_ids = output_ids[
        :, inputs["input_ids"].shape[1]:
    ]

    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return answer.strip()


# ========== 5. 分别测试两个模型 ==========
def evaluate_model(use_lora=False):
    label = "Fine-tuned" if use_lora else "Baseline"

    print(f"\n===== {label} =====")

    model = load_model(use_lora=use_lora)

    predictions = []

    for sample in tqdm(samples, desc=label):
        answer = predict(model, sample)
        predictions.append(answer)

    # 释放模型，避免两个模型同时占用显存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return predictions


baseline_preds = evaluate_model(use_lora=False)
finetuned_preds = evaluate_model(use_lora=True)


# ========== 6. 计算 BLEU-4 和 ROUGE-L ==========
references = [s["answer"] for s in samples]

scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True,
)


def calculate_metrics(predictions):

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
        tokenize="13a",
        #effective_order=True,
    ).score

    rouge_scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(
            score["rougeL"].fmeasure
        )

    rouge_l = sum(rouge_scores) / len(rouge_scores)

    return {
        "BLEU-4": bleu,
        "ROUGE-L": rouge_l * 100,
    }


baseline_metrics = calculate_metrics(baseline_preds)
finetuned_metrics = calculate_metrics(finetuned_preds)


# ========== 7. 保存逐样本预测 ==========
output_csv = OUTPUT_DIR / "qlora_predictions_short.csv"

with open(
    output_csv, "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "instruction",
        "reference",
        "baseline",
        "finetuned",
    ])

    for sample, baseline, finetuned in zip(
        samples,
        baseline_preds,
        finetuned_preds,
    ):
        writer.writerow([
            sample["image"],
            sample["instruction"],
            sample["answer"],
            baseline,
            finetuned,
        ])


# ========== 8. 保存汇总指标 ==========
metrics_csv = OUTPUT_DIR / "qlora_metrics_short.csv"

with open(
    metrics_csv, "w", newline="", encoding="utf-8"
) as f:

    writer = csv.writer(f)
    writer.writerow([
        "Metric",
        "Baseline",
        "Fine-tuned",
        "Change",
    ])

    for name in baseline_metrics:
        base = baseline_metrics[name]
        fine = finetuned_metrics[name]

        writer.writerow([
            name,
            round(base, 4),
            round(fine, 4),
            round(fine - base, 4),
        ])


# ========== 9. 打印结果 ==========
print("\n===== FINAL EVALUATION =====")

for name in baseline_metrics:
    base = baseline_metrics[name]
    fine = finetuned_metrics[name]

    print(
        f"{name}: "
        f"Baseline={base:.2f} | "
        f"Fine-tuned={fine:.2f} | "
        f"Change={fine-base:+.2f}"
    )


print("\n===== EXAMPLE PREDICTIONS =====")

for i in range(min(3, len(samples))):
    print(f"\nExample {i+1}")
    print("Instruction:", samples[i]["instruction"])
    print("Reference:", references[i])
    print("Baseline:", baseline_preds[i])
    print("Fine-tuned:", finetuned_preds[i])

print(f"\nSaved predictions: {output_csv}")
print(f"Saved metrics: {metrics_csv}")
print("Evaluation finished!")
