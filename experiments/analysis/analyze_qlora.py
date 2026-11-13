
import csv
import re
from pathlib import Path
from statistics import mean, median

INPUT = Path("results/qlora_predictions_short.csv")
OUTPUT = Path("results/qlora_length_analysis_short.csv")

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

def count_words(text):
    return len(re.findall(r"\b[\w']+\b", text))

def analyze(column):
    lengths = [count_words(row[column]) for row in rows]

    return {
        "mean": mean(lengths),
        "median": median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }

print("===== Caption Length Analysis =====")

for name, column in [
    ("Reference", "reference"),
    ("Baseline", "baseline"),
    ("Fine-tuned", "finetuned"),
]:
    result = analyze(column)

    print(
        f"{name}: "
        f"Mean={result['mean']:.2f} | "
        f"Median={result['median']} | "
        f"Min={result['min']} | "
        f"Max={result['max']}"
    )

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image",
        "reference_words",
        "baseline_words",
        "finetuned_words",
    ])

    for row in rows:
        writer.writerow([
            row["image"],
            count_words(row["reference"]),
            count_words(row["baseline"]),
            count_words(row["finetuned"]),
        ])

print(f"\nSaved to {OUTPUT}")
