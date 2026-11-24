import cv2
import numpy as np

LINE_HEIGHT = 64  # CRNN input height


def load_gray(path):
    # imdecode instead of imread - works with non-ASCII (Polish) paths on Windows
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def binarize(gray):
    # Gaussian blur against noise, adaptive threshold against uneven lighting of photos
    # text -> 255, background -> 0 (findContours looks for white objects)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    # remove specks of noise (tiny connected components)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 8
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def deskew(gray, binary, max_angle=5.0, step=0.25):
    # try small rotations - when text lines are horizontal the row profile has the sharpest peaks
    small = cv2.resize(binary, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    h, w = small.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rows = cv2.warpAffine(small, M, (w, h)).sum(axis=1, dtype=np.float64)
        score = float(np.sum(rows ** 2))
        if score > best_score:
            best_angle, best_score = angle, score
    if abs(best_angle) < step:
        return gray, binary
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return gray, binary


def clean_binary(gray, binary, max_ratio=0.6):
    # Mask used ONLY to find the lines (the gray image given to the network is not changed):
    # remove notebook lines, margins, page edges, the other page and shadows.
    h, w = binary.shape
    paper = cv2.dilate(gray, np.ones((25, 25), np.uint8))   # local paper brightness (max filter)
    binary = np.where(gray > max_ratio * paper, 0, binary).astype(np.uint8)  # too light to be pen ink

    def opened(img, kw, kh):  # long straight runs = printed lines, not handwriting
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
    lines = cv2.dilate(opened(binary, w // 15, 1) | opened(binary, 1, h // 30), np.ones((3, 3), np.uint8))
    binary = np.where(lines > 0, 0, binary).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights, areas = stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    char_h = float(np.median(heights[1:][areas[1:] > 30])) if np.any(areas[1:] > 30) else 20.0
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    right, bottom = left + stats[:, cv2.CC_STAT_WIDTH], top + heights
    m = int(0.01 * max(h, w))
    # blobs taller than several text lines (page edge, spine) or near the photo border (desk, other page)
    drop = (heights > 6 * char_h) | (left <= m) | (top <= m) | (right >= w - m) | (bottom >= h - m) | (areas < 8)
    drop[0] = False
    return np.where(drop[labels], 0, binary).astype(np.uint8), char_h


def find_lines(binary, char_h):
    # Text lines as horizontal bands: the row profile of ink has a peak in every line and a valley
    # between lines. Each band ends in the deepest valley, so the boxes can never overlap - letters
    # sticking into the neighbour band are cut off (like in IAM crops) and cleaned by normalize_line.
    profile = cv2.GaussianBlur(binary.sum(axis=1, dtype=np.float64).reshape(-1, 1), (1, 0), char_h / 3).ravel()
    window = max(3, int(char_h)) | 1
    is_peak = profile >= cv2.dilate(profile.reshape(-1, 1), np.ones((window, 1))).ravel()
    peaks = [i for i in np.flatnonzero(is_peak & (profile > 0.15 * profile.max()))]
    centers = []
    for p in peaks:                                   # merge peaks of one line (flat tops, ascenders)
        if centers and p - centers[-1] < 0.8 * char_h:
            if profile[p] > profile[centers[-1]]:
                centers[-1] = p
        else:
            centers.append(p)
    if not centers:
        return []
    cuts = [0] + [a + int(np.argmin(profile[a:b])) for a, b in zip(centers, centers[1:])] + [len(profile)]

    boxes = []
    for y0, y1 in zip(cuts, cuts[1:]):
        band = binary[y0:y1] > 0
        if band.sum() < char_h * char_h:                    # empty band or a few specks
            continue
        # horizontal extent: groups of ink columns separated by gaps > 3 char heights,
        # keep the groups with real text (lone specks at the page edge would stretch the box)
        cols = np.flatnonzero(band.any(axis=0))
        groups = np.split(cols, np.flatnonzero(np.diff(cols) > 3 * char_h) + 1)
        groups = [g for g in groups if band[:, g].sum() > 0.5 * char_h * char_h]
        if not groups:
            continue
        x0, x1 = int(groups[0][0]), int(groups[-1][-1]) + 1
        # vertical extent: the block of rows around the densest row (drops remnants far above / below)
        ink_rows = band[:, x0:x1].sum(axis=1)
        rows = np.flatnonzero(ink_rows)
        blocks = np.split(rows, np.flatnonzero(np.diff(rows) > 0.5 * char_h) + 1)
        rows = next(b for b in blocks if int(np.argmax(ink_rows)) in b)
        if rows[-1] - rows[0] + 1 < 0.6 * char_h:          # thin remnant of an edge or a line
            continue
        boxes.append((x0, y0 + int(rows[0]), x1 - x0, int(rows[-1] - rows[0]) + 1))
    return boxes


def crop_line(gray, box, char_h):
    # small horizontal margin; vertically the band itself (never reaches into the neighbour lines)
    x, y, w, h = box
    pad = int(0.3 * char_h)
    return gray[y:y + h, max(0, x - pad):x + w + pad]


def normalize_line(gray, height=LINE_HEIGHT):
    # Crop of a line / word sequence (uint8, dark text) -> float32 [height, W] in [0, 1].
    # Used for training data AND for inference, so the network always gets the same kind of input.
    # flatten the lighting (shadow of the hand / notebook spine): divide by the paper brightness,
    # estimated with a max filter wider than the pen strokes
    k = max(15, gray.shape[0] // 3)
    paper = cv2.GaussianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), (0, 0), k / 2)
    gray = np.clip(gray.astype(np.float32) / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)
    _, ink = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # remove pieces of neighbouring lines: small blobs cut by the top or bottom edge of the crop
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    top, hgt = stats[:, cv2.CC_STAT_TOP], stats[:, cv2.CC_STAT_HEIGHT]
    foreign = ((top == 0) | (top + hgt == ink.shape[0])) & (hgt < 0.4 * ink.shape[0])
    # and a thin stroke from the very top to the very bottom = page edge / fold, not a letter
    foreign |= (top == 0) & (hgt == ink.shape[0]) & (stats[:, cv2.CC_STAT_WIDTH] < 0.08 * ink.shape[0])
    foreign[0] = False
    gray[foreign[labels]] = int(np.percentile(gray, 90))
    ink[foreign[labels]] = 0

    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return np.ones((height, height // 2), dtype=np.float32)
    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)

    # contrast stretch: darkest ink -> 0, paper -> 1
    lo, hi = gray.min(), np.percentile(gray, 90)
    gray = np.clip((gray - lo) / max(hi - lo, 1.0), 0.0, 1.0)

    # fixed height (4 px margin top/bottom, 8 px left/right), width keeps the aspect ratio
    new_w = max(1, round(gray.shape[1] * (height - 8) / gray.shape[0]))
    shrink = gray.shape[0] > height - 8  # INTER_AREA only for shrinking, enlarging with it gives jagged strokes
    gray = cv2.resize(gray, (new_w, height - 8), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    gray = np.clip(gray, 0.0, 1.0)
    return cv2.copyMakeBorder(gray, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=1.0)


def augment(img, rng):
    # training only - makes clean IAM scans / rendered fonts look like phone photos of notes
    h, w = img.shape
    img = cv2.resize(img, (max(8, int(w * rng.uniform(0.8, 1.2))), h))     # wider / narrower writing
    w = img.shape[1]
    if rng.random() < 0.5:  # slant
        s = rng.uniform(-0.2, 0.2)
        img = cv2.warpAffine(img, np.float32([[1, s, -s * h / 2], [0, 1, 0]]), (w, h), borderValue=1.0)
    if rng.random() < 0.3:  # line not perfectly horizontal + imprecise segmentation (shifted up / down)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2.5, 2.5), 1.0)
        M[1, 2] += rng.uniform(-4, 4)
        img = cv2.warpAffine(img, M, (w, h), borderValue=1.0)
    if rng.random() < 0.4:  # elastic distortion: every writer draws letters a bit differently
        dx, dy = (cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 5) for _ in range(2))
        amp = rng.uniform(1.0, 2.5) / max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        img = cv2.remap(img, gx + dx * amp, gy + dy * amp, cv2.INTER_LINEAR, borderValue=1.0)
    if rng.random() < 0.3:  # low-resolution photo: shrink and enlarge back
        f = rng.uniform(0.35, 0.7)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.3:  # thinner / thicker pen (erode makes dark strokes thicker on white paper)
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    if rng.random() < 0.2:  # notebook ruling going through the text
        for y in rng.integers(0, h, size=rng.integers(1, 3)):
            cv2.line(img, (0, int(y)), (w - 1, int(y)), float(rng.uniform(0.55, 0.85)), int(rng.integers(1, 3)))
    if rng.random() < 0.4:  # gray / cream paper and a shadow across the line (multiplicative)
        light = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), w, dtype=np.float32)[None, :]
        img = img * light * rng.uniform(0.85, 1.0)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img + rng.normal(0, rng.uniform(0.0, 0.04), img.shape)
    if rng.random() < 0.3:  # JPEG compression of phone photos
        ok, buf = cv2.imencode(".jpg", np.clip(img * 255, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 80))])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0).astype(np.float32)
