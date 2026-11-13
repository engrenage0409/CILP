
import csv
from pathlib import Path

BASE = Path("results/docvqa/baseline_predictions.csv")
FINE = Path("results/docvqa/finetuned_predictions.csv")

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

baseline = read_csv(BASE)
finetuned = read_csv(FINE)

# 按图片 + 问题匹配，避免依赖行顺序
def key(row):
    return (row["image"], row["question"])

base_map = {key(row): row for row in baseline}
fine_map = {key(row): row for row in finetuned}

assert base_map.keys() == fine_map.keys(), (
    "Baseline and fine-tuned samples do not match"
)

improved = []
regressed = []
unchanged = []

for k in base_map:
    b = base_map[k]
    f = fine_map[k]

    base_anls = float(b["anls"])
    fine_anls = float(f["anls"])

    result = {
        "question": b["question"],
        "reference": b["reference"],
        "baseline": b["prediction"],
        "finetuned": f["prediction"],
        "baseline_anls": base_anls,
        "finetuned_anls": fine_anls,
        "change": fine_anls - base_anls,
    }

    if result["change"] > 0:
        improved.append(result)
    elif result["change"] < 0:
        regressed.append(result)
    else:
        unchanged.append(result)

print("===== DocVQA Error Analysis =====")
print("Improved:", len(improved))
print("Regressed:", len(regressed))
print("Unchanged:", len(unchanged))

regressed.sort(key=lambda r: r["change"])

print("\n===== Top 5 Regressions =====")

for row in regressed[:5]:
    print("\nQuestion:", row["question"])
    print("Reference:", row["reference"])
    print("Baseline:", row["baseline"])
    print("Fine-tuned:", row["finetuned"])
    print("ANLS Change:", f"{row['change']:+.3f}")

print("\n===== Top 5 Improvements =====")

improved.sort(key=lambda r: r["change"], reverse=True)

for row in improved[:5]:
    print("\nQuestion:", row["question"])
    print("Reference:", row["reference"])
    print("Baseline:", row["baseline"])
    print("Fine-tuned:", row["finetuned"])
    print("ANLS Change:", f"{row['change']:+.3f}")
