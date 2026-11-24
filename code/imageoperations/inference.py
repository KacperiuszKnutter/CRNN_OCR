import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from imageoperations.dataset import load_vocab
from imageoperations.preprocessing import (binarize, clean_binary, crop_line, deskew, find_lines, load_gray,
                                           normalize_line)
from model.crnn import CRNN
from train.metrics import ctc_greedy_decode


@torch.no_grad()
def recognize(model, img, idx_to_char, device):
    # one normalized line / word sequence [64, W] -> text
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return ctc_greedy_decode(model(x), idx_to_char)[0]


def read_page(model, gray, idx_to_char, device, return_crops=False):
    # page -> binarization -> deskew -> clean mask -> line bands (never overlapping) -> CRNN -> text
    binary = binarize(gray)
    gray, binary = deskew(gray, binary)
    clean, char_h = clean_binary(gray, binary)
    boxes = find_lines(clean, char_h)
    crops = [crop_line(gray, box, char_h) for box in boxes]
    lines = [recognize(model, normalize_line(crop), idx_to_char, device) for crop in crops]
    if return_crops:
        return "\n".join(lines), gray, boxes, crops, lines
    return "\n".join(lines), gray, boxes


def export_lines(folder, image_name, crops, texts, split):
    # line crops + labels.csv with the recognized text -> correct the text by hand, put the folder
    # into dataset/ and run mergeexcels.py: it becomes a new training source
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "labels.csv"
    new_file = not csv_path.exists()
    # utf-8 with BOM for a new file, so Excel shows Polish letters correctly
    with open(csv_path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["filepath", "text", "split"])
        for i, (crop, text) in enumerate(zip(crops, texts)):
            name = f"{Path(image_name).stem}_{i:02d}.png"
            cv2.imencode(".png", crop)[1].tofile(str(folder / name))
            writer.writerow([f"{folder.name}/{name}", text, split])
    print(f"Saved {len(crops)} lines to {folder}")


def main():
    parser = argparse.ArgumentParser(description="OCR: image -> text")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=str(CODE_DIR / "train" / "checkpoints_base32" / "best_crnn.pth"))
    parser.add_argument("--line", action="store_true", help="the image contains a single line of text")
    parser.add_argument("--debug", help="save the page with detected boxes to this file")
    parser.add_argument("--export", help="save line crops + labels.csv to this folder (for labeling)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"],
                        help="split written to labels.csv by --export")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, idx_to_char, num_classes = load_vocab(CODE_DIR / "configs" / "train_config.yaml")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = CRNN(num_chars=num_classes, base=ckpt.get("base", 16)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gray = load_gray(args.image)
    if args.line:
        print(recognize(model, normalize_line(gray), idx_to_char, device))
        return

    text, gray, boxes, crops, lines = read_page(model, gray, idx_to_char, device, return_crops=True)
    print(text)
    if args.export:
        export_lines(args.export, args.image, crops, lines, args.split)
    if args.debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(args.debug, vis)


if __name__ == "__main__":
    main()
