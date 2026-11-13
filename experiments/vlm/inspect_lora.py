
import torch
from transformers import (
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ========== 1. 基本配置 ==========
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ========== 2. 加载 4-bit 模型 ==========
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# ========== 3. 查看 Attention 层 ==========
print("\n===== Attention Projection Layers =====")

# 找到语言模型中的 q_proj / v_proj
# 避免误将 LoRA 插入 Vision Encoder
target_names = []

for name, module in model.named_modules():

    is_language_layer = (
        "language_model" in name
        and ".layers." in name
    )

    is_target = name.endswith(
        (".q_proj", ".v_proj")
    )

    if is_language_layer and is_target:
        print(name)
        target_names.append(name)

if not target_names:
    raise RuntimeError(
        "No language-model q_proj/v_proj found. "
        "Inspect model.named_modules() before adding LoRA."
    )

print(f"\nFound {len(target_names)} target modules")

# ========== 4. 准备量化模型训练 ==========
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

# ========== 5. 配置 LoRA ==========
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_names,
    bias="none",
    task_type="CAUSAL_LM",
)

# ========== 6. 注入 LoRA ==========
model = get_peft_model(
    model,
    lora_config
)

# ========== 7. 查看可训练参数 ==========
print("\n===== Trainable Parameters =====")

model.print_trainable_parameters()

# ========== 8. 查看 LoRA 注入位置 ==========
print("\n===== LoRA Modules =====")

count = 0

for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        print(name)
        count += 1

print(f"\nTotal LoRA-adapted modules: {count}")

# ========== 9. 检查是否误训练视觉模块 ==========
vision_trainable = [
    name
    for name, param in model.named_parameters()
    if param.requires_grad and "visual" in name
]

print("\nTrainable parameters in visual modules:")
print(vision_trainable)

assert not vision_trainable, (
    "Unexpected trainable visual parameters!"
)

print("\nQLoRA adapter initialization successful!")
