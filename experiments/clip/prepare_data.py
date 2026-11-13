
from pathlib import Path              # 处理文件和目录
import csv                            # 写入 CSV 文件
from datasets import load_dataset     # 下载 Hugging Face 数据集

dataset = load_dataset("intro/flickr8k") # 加载 Flickr8k
image_dir = Path("images")             # 图片保存目录
data_dir = Path("data")                # CSV 保存目录

image_dir.mkdir(exist_ok=True)         # 创建 images 文件夹
data_dir.mkdir(exist_ok=True)          # 创建 data 文件夹

split_names = {                        # Hugging Face 与本地名称对应关系
    "train": "train",                  # 训练集
    "dev": "val",                      # 验证集
    "test": "test"                     # 测试集
}

for source, target in split_names.items():
    output_csv = data_dir / f"{target}.csv"  # CSV 输出路径

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)          # 创建 CSV Writer
        writer.writerow(["image", "caption"]) # CSV 表头

        for index, sample in enumerate(dataset[source]):
            filename = f"{target}_{index:05d}.jpg" # 唯一图片名
            image = sample["image"].convert("RGB") # 获取 RGB 图片
            image.save(image_dir / filename, quality=95) # 保存图片

            caption = sample["caption_0"] # 先选第一条 Caption
            writer.writerow([filename, caption]) # 保存图文配对

    print(f"{target}: {len(dataset[source])} pairs saved")
