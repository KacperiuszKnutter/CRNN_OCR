import csv
import sys
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from imageoperations.preprocessing import augment, load_gray, normalize_line

def _prepare(args):
    # one line -> normalized uint8 image (module level function, so multiprocessing can pickle it)
    path, target_h = args
    return np.round(normalize_line(load_gray(path), target_h) * 255).astype(np.uint8)

def load_vocab(configPath):
    with open(configPath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    classes = config['classes']
    class_dictionary = {ch: idx + 1 for idx, ch in enumerate(classes)}  # 0 = CTC blank
    idx_to_char = {idx + 1: ch for idx, ch in enumerate(classes)}
    return class_dictionary, idx_to_char, config["num_of_classes"]

# needed to pack batches in the pytorch's Dl it takes the arr of samples returned through OCRDataset.__getitem__ and transforms them to format required by torch.nn.CTCLoss
def ctc_collate_fn(batch):
    batch_size = len(batch)
    max_width = max(item[0].shape[2] for item in batch)
    # white bg pad
    padded_images = torch.ones((batch_size, 1, 64, max_width), dtype=torch.float32)
    input_lengths = torch.zeros(batch_size, dtype=torch.int32)
    target_lengths = torch.zeros(batch_size, dtype=torch.int32)
    target_list = []
    raw_texts = []
    sources = []
    for i, (img_tensor, target_tensor, raw_text, source) in enumerate(batch):
        w = img_tensor.shape[2]
        padded_images[i, :, :, :w] = img_tensor

        # crnn reduced width by a factor of 4
        input_lengths[i] = w //4
        target_lengths[i] = len(target_tensor)
        target_list.append(target_tensor)
        raw_texts.append(raw_text)
        sources.append(source)

    targets = torch.cat(target_list)
    return padded_images, targets, input_lengths, target_lengths, raw_texts, sources

class OCRDataset(Dataset):
    def __init__(self, csvPath: str | Path, dataset_root: str | Path, char_to_idx: dict[str, int], split: str = "train", target_h : int = 64, augment: bool = False, preload: bool = False):
        self.dataset_root = dataset_root
        self.char_to_idx = char_to_idx
        self.target_h = target_h
        self.augment = augment
        self.samples = []

        with open(csvPath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    img_path = self.dataset_root / row["filepath"]
                    # skip lines with chars outside the alphabet (#, &, * in IAM) - label wouldn't match the image
                    if any(ch not in self.char_to_idx for ch in row["text"]):
                        continue
                    encoded_text = [self.char_to_idx[ch] for ch in row["text"]]
                    if encoded_text and img_path.exists():
                        source = row["filepath"].split("/")[0]  # handwritten / printed
                        self.samples.append((img_path, encoded_text, row["text"], source))

        # preload: normalize every line once (all CPU cores) and keep it in RAM (~1 GB for the train split),
        # so epochs no longer wait for PNG decoding + preprocessing - only augmentation stays per batch
        self.images = None
        if preload:
            with Pool() as pool:
                self.images = pool.map(_prepare, [(s[0], target_h) for s in self.samples], chunksize=64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, encoded_text, text, source = self.samples[idx]

        # same preprocessing as in inference: crop to ink, contrast stretch, height 64
        if self.images is not None:
            img = self.images[idx].astype(np.float32) / 255.0
        else:
            img = normalize_line(load_gray(img_path), self.target_h)
        if self.augment:
            img = augment(img, np.random.default_rng())

        # CTC needs at least len(text) + 1 time steps, the CRNN reduces width by a factor of 4
        min_w_for_ctc = (len(encoded_text) + 1) * 4
        if img.shape[1] < min_w_for_ctc:
            img = cv2.resize(img, (min_w_for_ctc, self.target_h))

        # [1, 64, W] in [0, 1]
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        target_tensor = torch.tensor(encoded_text, dtype=torch.long)

        return img_tensor, target_tensor, text, source
