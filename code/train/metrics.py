import torch


def ctc_greedy_decode(log_probs: torch.Tensor, idx_to_char: dict[int, str], blank_idx: int = 0, lengths=None) -> list[str]:
    """
    Decodes CRNN log-probabilities [T, B, num_classes] into strings using greedy CTC decoding:
    1. Argmax at each time step
    2. Collapse consecutive duplicate indices
    3. Remove blank tokens (index 0)
    """
    # [T, B, C] -> [T, B] -> [B, T]
    best_paths = torch.argmax(log_probs, dim=2).transpose(0, 1).cpu().tolist()

    # ignore frames that come from the white padding added in the batch
    if lengths is not None:
        best_paths = [path[:int(n)] for path, n in zip(best_paths, lengths)]

    decoded_texts = []
    for path in best_paths:
        chars = []
        prev_idx = None
        for idx in path:
            if idx != prev_idx:
                if idx != blank_idx and idx in idx_to_char:
                    chars.append(idx_to_char[idx])
            prev_idx = idx
        decoded_texts.append("".join(chars))

    return decoded_texts


def levenshtein_distance(s1: str, s2: str) -> int:
    #Computes character-level Levenshtein edit distance between two strings
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class OCRMetrics:
    #Accumulates Character Error Rate (CER) and exact Line Accuracy across batches
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_edits = 0
        self.total_chars = 0
        self.exact_matches = 0
        self.total_samples = 0

    def update(self, predictions: list[str], targets: list[str]):
        for pred, target in zip(predictions, targets):
            self.total_edits += levenshtein_distance(pred, target)
            self.total_chars += max(1, len(target))
            if pred == target:
                self.exact_matches += 1
            self.total_samples += 1

    def compute(self) -> dict[str, float]:
        cer = self.total_edits / max(1, self.total_chars)
        line_acc = self.exact_matches / max(1, self.total_samples)
        return {
            "cer": cer,
            "line_accuracy": line_acc,
        }