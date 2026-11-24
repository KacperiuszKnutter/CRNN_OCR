import os 
import random 
import unicodedata
import cv2 
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import yaml
import csv 
from pathlib import Path
import re



def download_config():
    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed_chars = set(config["classes"])
    return allowed_chars

def font_supports_text(font_path, text):
    try:
        # check whether the font supports all characters in the text
        from fontTools.ttLib import TTFont
        tt = TTFont(font_path)
        cmap = tt.getBestCmap()
        return all(ord(ch) in cmap for ch in text if ch != " ")
    except Exception as e:
        print(f"Error checking font support for {font_path}: {e}")
        return True # fallback to True if there's an error checking the font

def add_noise_to_image(image_np):
    h, w = image_np.shape

    # add random angle rotation
    angle = random.uniform(-1.5, 1.5)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image_np = cv2.warpAffine(image_np, M, (w, h), borderValue=255)

    if random.random() < 0.4:
        # add random thickness to the text
        kernel = np.ones((2,2), np.uint8)
        if random.random() < 0.5:
            image_np = cv2.erode(image_np, kernel, iterations=1) # thicker text
        else:
            image_np = cv2.dilate(image_np, kernel, iterations=1) # thinner text

    if random.random() < 0.5:
        image_np = cv2.GaussianBlur(image_np, (3, 3), 0) # add Gaussian blur

    noise = np.random.normal(0, random.uniform(2,7), image_np.shape)
    image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
    return image_np

# fin - pdf
def render_image(text, font_path, target_height=64, padding=10):
    text = unicodedata.normalize("NFC", text)
    font = ImageFont.truetype(font_path, size=48)

    
    dummy_img = Image.new("L", (1, 1), color=255)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]


    bg_color = random.randint(235, 255)
    fg_color = random.randint(0, 50)
    pad_x, pad_y = random.randint(10, 25), random.randint(8, 16)

    img = Image.new("L", (text_w + 2 * pad_x, text_h + 2 * pad_y), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=fg_color)

    img_np = np.array(img)
    img_np = add_noise_to_image(img_np)

    h, w = img_np.shape
    new_w = max(16, int(w * (target_height / h)))
    img_resized = cv2.resize(img_np, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return img_resized

def get_valid_fonts(fonts_dir, allowed_chars):

    # only returns the fonts that support whole alphabet
    valid = []
    test_str = "".join(allowed_chars)
    for font_path in Path(fonts_dir).rglob("*.[tToO][tTfF][fF]"):
        if font_supports_text(str(font_path), test_str):
            valid.append(str(font_path))
    return valid

def detokenize(text):
    # undo WikiText/IAM tokenization: "Ben , who" -> "Ben, who", "( 25 mm )" -> "(25 mm)", "Nobody 's" -> "Nobody's"
    text = re.sub(r" ?@([-,.])@ ?", r"\1", text)
    text = re.sub(r" +([.,;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r" (n't|'s|'ll|'re|'ve|'d|'m)\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_lines_from_books(books_dir, allowed_chars, min_len=25, max_len=55):
    # slices books int .txt format to a natural lines of text
    lines = []
    replacements = {"„": '"', "”": '"', "’": "'", "‘": "'", "–": "-", "—": "-", "…": "..."}

    for txt_path in Path(books_dir).rglob("*.txt"):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = unicodedata.normalize("NFC", text)
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = detokenize(text)  # clean BEFORE rendering, so the label always matches the image
        text = "".join(ch for ch in text if ch in allowed_chars)
        words = text.split(" ")

        current_line = []
        current_len = 0
        target_len = random.randint(min_len, max_len)

        for word in words:
            if not word:
                continue
            if current_len + len(word) + 1 <= target_len:
                current_line.append(word)
                current_len += len(word) + 1
            else:
                if current_len >= min_len:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
                target_len = random.randint(min_len, max_len)
        if current_len >= min_len:
            lines.append(" ".join(current_line))

    random.shuffle(lines)
    return lines

def generate_numeric_lines(count=2500):
    # generates lines with dates codes and prices
    samples = []
    prefixes = ["FV", "NR", "ID", "NIP", "TEL", "KOD", "REF", "ORD", "PL", "EN"]
    for _ in range(count):
        parts = [
            f"{random.randint(1, 31):02d}.{random.randint(1, 12):02d}.{random.randint(1990, 2030)}",
            f"{random.choice(prefixes)}/{random.randint(100, 9999)}/{random.randint(20, 26)}",
            f"{random.randint(10, 99)}-{random.randint(100, 999)}",
            f"{random.randint(1, 9999)},{random.randint(0, 99):02d}",
            f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(1000, 99999)}",
        ]
        random.shuffle(parts)
        samples.append(" ".join(parts[:random.randint(2, 4)]))
    return samples

if __name__ == "__main__":
    allowed_chars = download_config()
    fonts = get_valid_fonts("fonts/", allowed_chars)
    if not fonts:
        raise RuntimeError("Brak poprawnych plików .ttf/.otf w ../dataset/fonts!")

    # Download ~6000 lines z PL i EN + 2500 digital lines  = 14 500 images
    pl_lines = extract_lines_from_books("../dataset/books/pl", allowed_chars)[:6000]
    # some lines in uppercase - otherwise Ć, Ę, Ń never appear in the data
    pl_lines = [line.upper() if random.random() < 0.05 else line for line in pl_lines]
    en_lines = extract_lines_from_books("../dataset/books/eng", allowed_chars)[:6000]
    num_lines = generate_numeric_lines(2500)

    all_texts = pl_lines + en_lines + num_lines
    random.shuffle(all_texts)

    out_dir = Path("../dataset/printed")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(all_texts)
    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "text", "split"])

        for idx, text in enumerate(all_texts):
            # split 85% train / 10% validation / 5% test
            if idx < int(0.85 * total):
                split = "train"
            elif idx < int(0.95 * total):
                split = "validation"
            else:
                split = "test"

            font_path = random.choice(fonts)
            img = render_image(text, font_path, target_height=64)

            filename = f"printed_{idx:05d}.png"
            cv2.imwrite(str(out_dir / filename), img)
            writer.writerow([f"printed/{filename}", text, split])

    print(f"Generated {total} imgs in {out_dir} with labels.csv.")