import csv
import re
from pathlib import Path



csv_path_root = Path("../dataset/labels.csv")

train_rows = []
validation_rows = []
test_rows = []



with open(csv_path_root, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        split = row["split"].strip().lower()
        if split == "train":
            train_rows.append(row)
        elif split == "validation":
            validation_rows.append(row)
        elif split == "test":
            test_rows.append(row)
        else:
            raise ValueError(f"Unknown split: {split}")

print(f"Number of rows in each split:")
print(f"  Train: {len(train_rows)}; percentage: {len(train_rows) / (len(train_rows) + len(validation_rows) + len(test_rows)) * 100:.2f}%")
print(f"  Validation: {len(validation_rows)}; percentage: {len(validation_rows) / (len(train_rows) + len(validation_rows) + len(test_rows)) * 100:.2f}%")
print(f"  Test: {len(test_rows)}; percentage: {len(test_rows) / (len(train_rows) + len(validation_rows) + len(test_rows)) * 100:.2f}%")
