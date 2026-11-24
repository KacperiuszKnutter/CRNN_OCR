import argparse
import math
import random
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no display on Colab / servers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

# Add code/ directory to sys.path for clean imports
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from imageoperations.dataset import OCRDataset, ctc_collate_fn, load_vocab
from model.crnn import CRNN
from train.metrics import OCRMetrics, ctc_greedy_decode


class LengthBucketSampler(Sampler):
    # Batches of lines with similar text length (= similar image width) -> much less white padding.
    # Small random noise in the sort key + shuffled batch order keep the batches different every epoch.
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths, self.batch_size, self.shuffle = lengths, batch_size, shuffle

    def __iter__(self):
        noise = (lambda: random.uniform(0, 4)) if self.shuffle else (lambda: 0)
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i] + noise())
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


def make_loader(dataset, batch_size, shuffle, workers):
    lengths = [len(sample[1]) for sample in dataset.samples]
    return DataLoader(dataset, batch_sampler=LengthBucketSampler(lengths, batch_size, shuffle),
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, collate_fn=ctc_collate_fn)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for step, (images, targets, input_lengths, target_lengths, _, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass in mixed precision (float16 on GPU) -> [T, B, num_classes]
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            log_probs = model(images)
        # CTC is numerically unstable in float16 -> always compute it in float32
        loss = criterion(log_probs.float(), targets, input_lengths, target_lengths)

        # Backward pass + gradient clipping for BiLSTM stability (unscale first when using AMP)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if step % 100 == 0 or step == len(loader):
            print(f"  Step [{step}/{len(loader)}] - Batch Loss: {loss.item():.4f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, idx_to_char, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    metrics = OCRMetrics()
    source_metrics = {}  # CER separately for handwritten / printed
    sample_examples = []

    for images, targets, input_lengths, target_lengths, raw_texts, sources in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            log_probs = model(images)
        log_probs = log_probs.float()
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        running_loss += loss.item()

        preds = ctc_greedy_decode(log_probs, idx_to_char, blank_idx=0, lengths=input_lengths)
        metrics.update(preds, raw_texts)
        for pred, gt, source in zip(preds, raw_texts, sources):
            source_metrics.setdefault(source, OCRMetrics()).update([pred], [gt])

        # Save a few examples from the first batch to inspect progress
        if not sample_examples:
            for gt, pr in zip(raw_texts[:3], preds[:3]):
                sample_examples.append((gt, pr))

    results = metrics.compute()
    results["loss"] = running_loss / max(1, len(loader))
    results["cer_per_source"] = {s: m.compute()["cer"] for s, m in source_metrics.items()}
    return results, sample_examples


def save_training_plots(history: dict, output_path: Path):
    """Saves Loss and CER curves using Matplotlib."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train CTC Loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val CTC Loss", marker="o")
    ax1.set_title("CTC Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, [c * 100 for c in history["val_cer"]], label="Val CER (%)", color="tab:orange", marker="o")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Line Accuracy (%)", color="tab:green", marker="o")
    ax2.set_title("Validation OCR Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Percentage (%)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN + CTC")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15, help="early stopping: epochs without Val CER improvement")
    parser.add_argument("--dataset-dir", default=str(CODE_DIR.parent / "dataset"))
    parser.add_argument("--ckpt-dir", default=str(CODE_DIR / "train" / "checkpoints_base32"),
                        help="on Colab point this to Google Drive, so checkpoints_base32 survive a disconnect")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--base", type=int, default=16, help="network width: 16 = current model, 32 = 2x wider (train from scratch)")
    parser.add_argument("--no-preload", action="store_true", help="do not keep normalized lines in RAM")
    parser.add_argument("--finetune", help="start from the weights of this checkpoint (e.g. best_crnn.pth); use a new --ckpt-dir")
    parser.add_argument("--oversample", nargs="*", default=[], metavar="SOURCE=K",
                        help="show lines of a dataset folder K times per epoch, e.g. moje_pismo=10 siostra=10")
    return parser.parse_args()


def main():
    args = parse_args()
    config_file = CODE_DIR / "configs" / "train_config.yaml"
    dataset_dir = Path(args.dataset_dir)
    labels_file = dataset_dir / "labels.csv"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / "best_crnn.pth", ckpt_dir / "last_crnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Using device: {device} | mixed precision: {use_amp}")

    # --- Load Vocabulary & Datasets ---
    char_to_idx, idx_to_char, num_classes = load_vocab(config_file)

    train_ds = OCRDataset(labels_file, dataset_dir, char_to_idx, split="train", augment=True, preload=not args.no_preload)
    val_ds = OCRDataset(labels_file, dataset_dir, char_to_idx, split="validation", preload=not args.no_preload)
    # own handwriting is a few hundred lines next to ~19k others - repeat it, so the model really sees it
    for spec in args.oversample:
        source, k = spec.split("=")
        idx = [i for i, sample in enumerate(train_ds.samples) if sample[3] == source]
        train_ds.samples += [train_ds.samples[i] for i in idx] * (int(k) - 1)
        if train_ds.images is not None:
            train_ds.images += [train_ds.images[i] for i in idx] * (int(k) - 1)
        print(f"Oversampling {source}: {len(idx)} lines x {k}")
    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, workers=args.workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, workers=args.workers)

    # --- Model, Loss, Optimizer ---
    model = CRNN(num_chars=num_classes, in_channels=1, activation="leaky_relu", dropout=0.2, base=args.base).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (base={args.base})")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_cer, epochs_without_improvement = 1, float("inf"), 0
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_acc": []}

    # --- Fine-tuning: weights of an already trained model, fresh optimizer (only for a new run) ---
    if args.finetune and not last_path.exists():
        ckpt = torch.load(args.finetune, map_location=device)
        if ckpt.get("base", 16) != args.base:
            raise SystemExit(f"--base {args.base} does not match the checkpoint (base={ckpt.get('base', 16)})")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Fine-tuning from {args.finetune}")

    # --- Resume after a Colab disconnect ---
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch, best_cer = ckpt["epoch"] + 1, ckpt["best_cer"]
        epochs_without_improvement, history = ckpt["epochs_without_improvement"], ckpt["history"]
        print(f"Resumed from {last_path} (epoch {ckpt['epoch']}, best Val CER {best_cer * 100:.2f}%)")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no Val CER improvement for {args.patience} epochs")
            break
        start_time = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} (lr {optimizer.param_groups[0]['lr']:.2e}) ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics, examples = validate(model, val_loader, criterion, idx_to_char, device, use_amp)

        val_loss = val_metrics["loss"]
        val_cer = val_metrics["cer"]
        val_acc = val_metrics["line_accuracy"]

        scheduler.step(val_cer)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} finished in {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer * 100:.2f}% | Exact Line Acc: {val_acc * 100:.2f}%"
        )

        print("   CER per source: " + ", ".join(f"{s}: {c * 100:.2f}%" for s, c in val_metrics["cer_per_source"].items()))
        for gt, pr in examples:
            print(f"   GT: '{gt}'")
            print(f"   PR: '{pr}'")

        if val_cer < best_cer:
            best_cer, epochs_without_improvement = val_cer, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_cer": val_cer, "base": args.base}, best_path)
            print(f"  --> Saved new best model (Val CER: {best_cer * 100:.2f}%)")
        else:
            epochs_without_improvement += 1

        # full training state after every epoch -> training can be resumed from here
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_cer": best_cer,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }, last_path)
        save_training_plots(history, ckpt_dir / "training_curves.png")

    # --- Final evaluation of the best model on the test split ---
    test_ds = OCRDataset(labels_file, dataset_dir, char_to_idx, split="test", preload=not args.no_preload)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, workers=args.workers)
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    test_metrics, _ = validate(model, test_loader, criterion, idx_to_char, device, use_amp)
    print(f"\nTEST ({len(test_ds)} lines): CER {test_metrics['cer'] * 100:.2f}% | "
          f"Line Acc {test_metrics['line_accuracy'] * 100:.2f}% | per source: {test_metrics['cer_per_source']}")


if __name__ == "__main__":
    main()
