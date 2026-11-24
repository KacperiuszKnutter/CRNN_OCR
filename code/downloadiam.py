import csv 
import os 
from datasets import load_dataset

dataset = load_dataset("Teklia/IAM-line")

output_dir = "../dataset/handwritten"

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "labels.csv"), mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # split from hugging face dataset into train, validation, and test sets
    writer.writerow(["image_path", "text", "split"])

    for split_name in ["train", "validation", "test"]:
        for idx, sample in enumerate(dataset[split_name]):
            image_path = os.path.join(output_dir, f"{split_name}_{idx:05d}.png")
            
            sample["image"].convert("L").save(image_path)
            writer.writerow([image_path, sample["text"], split_name])
