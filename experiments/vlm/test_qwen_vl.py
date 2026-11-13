
import torch
from PIL import Image
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig
)

# 1. 模型名称
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# 2. 配置 4-bit 量化
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. 加载预训练模型
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    dtype=torch.bfloat16
)

# 4. 加载配套图文处理器
processor = AutoProcessor.from_pretrained(MODEL_NAME)

model.eval()

# 5. 读取一张本地图片
image = Image.open("test.jpg").convert("RGB")

# 6. 构建多模态对话
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Describe this image in detail."
            }
        ]
    }
]

# 7. 把对话转换为模型所需的 Prompt
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 8. 将图像和文本转换为 Tensor
inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt"
).to(model.device)

# 9. 关闭梯度，进行推理
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )

# 10. 只保留新生成的答案
generated_ids = output_ids[
    :, inputs["input_ids"].shape[1]:
]

answer = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("Model Answer:")
print(answer)

# 11. 查看实际 CUDA 显存峰值
if torch.cuda.is_available():
    print(
        "Peak allocated VRAM (GB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    )
