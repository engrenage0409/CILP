
import csv
import re
from pathlib import Path
from statistics import mean

INPUT = Path("results/qlora_clipscore.csv")
OUTPUT = Path("results/qlora_error_analysis.csv")

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))


# ========== 1. 数据预处理 ==========
def word_count(text):
    return len(re.findall(r"\b[\w']+\b", text))


for row in rows:
    row["baseline_clipscore"] = float(
        row["baseline_clipscore"]
    )
    row["finetuned_clipscore"] = float(
        row["finetuned_clipscore"]
    )
    row["clipscore_change"] = float(
        row["clipscore_change"]
    )

    row["baseline_words"] = word_count(row["baseline"])
    row["finetuned_words"] = word_count(row["finetuned"])

    row["length_change"] = (
        row["finetuned_words"] - row["baseline_words"]
    )


# ========== 2. 统计分数变化 ==========
improved = [
    r for r in rows if r["clipscore_change"] > 0
]

declined = [
    r for r in rows if r["clipscore_change"] < 0
]

unchanged = [
    r for r in rows if r["clipscore_change"] == 0
]

print("\n===== Overall Statistics =====")
print(f"Total samples: {len(rows)}")
print(f"CLIPScore improved: {len(improved)}")
print(f"CLIPScore declined: {len(declined)}")
print(f"Unchanged: {len(unchanged)}")

print(
    "Mean CLIPScore change:",
    round(mean(r["clipscore_change"] for r in rows), 4)
)


# ========== 3. 按分数变化排序 ==========
sorted_rows = sorted(
    rows,
    key=lambda r: r["clipscore_change"]
)


def print_cases(title, cases):

    print(f"\n===== {title} =====")

    for i, row in enumerate(cases, 1):

        print(f"\nCase {i}")
        print("Image:", row["image"])
        print("Instruction:", row["instruction"])
        print("Reference:", row["reference"])

        print("\nBaseline:")
        print(row["baseline"])

        print("\nFine-tuned:")
        print(row["finetuned"])

        print(
            "\nCLIPScore:",
            f"{row['baseline_clipscore']:.4f}",
            "→",
            f"{row['finetuned_clipscore']:.4f}"
        )

        print(
            "Change:",
            f"{row['clipscore_change']:+.4f}"
        )

        print(
            "Word Count:",
            row["baseline_words"],
            "→",
            row["finetuned_words"]
        )


# 降幅最大的5张
print_cases(
    "Top 5 Largest Declines",
    sorted_rows[:5]
)

# 提升最大的5张
print_cases(
    "Top 5 Largest Improvements",
    sorted_rows[-5:][::-1]
)


# ========== 4. 保存分析结果 ==========
with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(sorted_rows)

print(f"\nSaved: {OUTPUT}")
