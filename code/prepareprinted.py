from pathlib import Path
import re
import unicodedata
import yaml

# load class config from train_config.yaml
with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

allowed_chars = set(config["classes"])
allowed_chars.add(" ")  # space as a seperator between words, but not at the beginning or end of a line

# Opcjonalne mapowanie znaków typograficznych z książek na ich odpowiedniki w configu
CHAR_REPLACEMENTS = {
    "„": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "’": "'",
    "‘": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    "@": " ",
}

def clean_text(raw_text):
    # Norm Unicode 
    text = unicodedata.normalize("NFC", raw_text)

    # replace typographic characters with their simpler equivalents
    for old_char, new_char in CHAR_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)

    # newline to space
    text = re.sub(r"\s+", " ", text)

    # delete characters not in allowed_chars
    cleaned = "".join(ch for ch in text if ch in allowed_chars)

    # again clear multiple spaces and trim
    return re.sub(r"\s+", " ", cleaned).strip()

# go through all .txt files in the specified folders and clean them
folders = [Path("../dataset/books/eng"), Path("../dataset/books/pl")]
processed_count = 0

for folder in folders:
    if not folder.exists():
        print(f"skipping: {folder}")
        continue

    # rglob("*.txt") searches for all .txt files in the folder and its subfolders
    for txt_path in folder.rglob("*.txt"):
        raw_content = txt_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_content = clean_text(raw_content)

        # ovverwrite the original file with the cleaned content
        txt_path.write_text(cleaned_content, encoding="utf-8")
        processed_count += 1
        print(f"Cleared: {txt_path} (chars: {len(cleaned_content)})")

print(f"\n Finished processing: {processed_count}")