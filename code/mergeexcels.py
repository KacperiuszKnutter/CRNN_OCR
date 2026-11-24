import csv
import re
from pathlib import Path

from createprintedpngs import detokenize

dataset_root = Path("../dataset")
merged_rows = []

# every folder in dataset/ with its own labels.csv is a source (handwritten = IAM, printed, own photos...)
for subfolder in sorted(p.name for p in dataset_root.iterdir() if (p / "labels.csv").exists()):
    csv_path = dataset_root / subfolder / "labels.csv"
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        # find a seperator
        sample = f.read(1024)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        print(f"Kolumny w {subfolder}/labels.csv: {reader.fieldnames}")

        # get the first col no matter the name
        path_col = reader.fieldnames[0]
        text_col = "text" if "text" in reader.fieldnames else reader.fieldnames[1]
        split_col = "split" if "split" in reader.fieldnames else (
            reader.fieldnames[2] if len(reader.fieldnames) > 2 else None
        )

        for row in reader:
            filename = Path(row[path_col].replace("\\", "/")).name
            rel_path = f"{subfolder}/{filename}"

            # get rid of additional white chars
            # printed labels must stay exactly as rendered on the image, IAM labels are tokenized ("Ben , who")
            text = detokenize(row[text_col]) if subfolder == "handwritten" else row[text_col].strip()
            split = row[split_col].strip() if split_col and row.get(split_col) else "train"

            if text:
                merged_rows.append([rel_path, text, split])

with open(dataset_root / "labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "text", "split"])
    writer.writerows(merged_rows)

print(f"Done! Summarized num of lines: {len(merged_rows)}")