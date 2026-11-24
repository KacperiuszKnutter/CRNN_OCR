# OCR – image to text (printed and handwritten, PL/EN)

[English](#english) | [Polski](#polski)

![Handwriting example](photos/Test_pismo_odreczne_1.png)

---

## English

The program converts a photo or scan of a page into text. It recognizes **printed text and handwriting (cursive)** in **Polish and English**. The model alphabet has 93 characters: Polish and English letters, digits, punctuation and space.

### How it works

The pipeline has two stages: **text line extraction** (OpenCV/NumPy) and **line recognition** with a neural network.

#### 1. Text extraction (`code/imageoperations/preprocessing.py`)

1. **Binarization:** Gaussian blur and adaptive thresholding (robust to shadows), then small noise removal.
2. **Deskewing:** horizontal projection profile for angles within ±5°. The angle where text lines are most horizontal is chosen.
3. **Mask cleaning:** removal of notebook ruling, margins, page edges and shadows. Criteria: brightness relative to the paper and long straight segments (morphological operations).
4. **Line segmentation:** each text line forms a peak in the horizontal profile. The boundary is placed at the deepest valley between lines, so **line crops never overlap**.
5. **Line normalization:** lighting correction, removal of fragments from neighbouring lines, cropping to the ink and scaling to 64 px height. The same function is used in training and in recognition.

#### 2. Recognition (`code/model/crnn.py`)

A **CRNN + CTC** network built from scratch in PyTorch, without pretrained weights:
- **Convolutional part:** 6 residual blocks, i.e. 16 convolutional layers with BatchNorm, LeakyReLU and Dropout2d. Downsampling uses convolution stride, no pooling.
- **Recurrent part:** 2-layer bidirectional LSTM (256 units per direction).
- **Output:** linear layer with 94 classes (93 characters + CTC blank).
- **Size:** about 4.7 M parameters (`--base 32` variant).

The network reads a whole line at once. Thanks to the CTC loss it learns by itself where each letter is, so the image does not have to be split into single characters, which does not work for cursive. Greedy CTC decoding and the CER metric (Levenshtein distance) are implemented by hand (`code/train/metrics.py`).

### Data and training

**Training set:**
- **IAM-line:** 10,373 lines of English handwriting, official split with disjoint writers in the test set.
- **Synthetic print:** 14,500 lines generated from Polish and English texts, 43 typefaces, distortions.
- **Own handwriting:** 243 hand-labeled lines of Polish notes (2 writers), phone photos.

Training uses augmentation that makes the data resemble phone photos: elastic distortions, slant, shadow and grey paper, notebook ruling, JPEG and low resolution.

**Training was done in Google Colab** (`notebooks/colab_train.ipynb`):

| Stage | Epochs | Val CER | Val line accuracy |
|---|---|---|---|
| Base training (IAM + print) | 34 | ≈ 2.7% | ≈ 66% |
| Fine-tuning on own handwriting (`--finetune`, lr 1e-4, own lines ×10) | 40 | ≈ 2.5% | ≈ 67% |

CER is computed on the combined validation set (print + IAM + own handwriting). The training log also reports CER per source. Training curves: [base](photos/training_curves_epochs_40.png), [fine-tuning](photos/training_curves_finetune.png).

### Running

Python 3.10+ is required.

```bash
pip install -r code/requirements.txt
```

On Windows with an NVIDIA GPU install PyTorch with CUDA, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu128`. The CPU version is enough for the GUI alone.

**Trained models (`*.pth`) are not in the repository.** Put `best_crnn.pth` in `code/train/checkpoints_finetune/`. If it is missing, the GUI will ask for the file at startup.

#### GUI

```bash
python GUI/gui.py
```

Choose an image (`.png`/`.jpg`). The program marks the detected lines, shows the recognized text for manual correction and saves it as a PDF note.

#### Command line (from `code/`)

```bash
python imageoperations/inference.py --image page.jpg --checkpoint train/checkpoints_finetune/best_crnn.pth --debug boxes.png
python imageoperations/inference.py --image line.png --line      # image with a single line
```

### Data preparation and training (from `code/`)

The `dataset/` folder is not in the repository. It is created by the scripts:

```bash
python downloadiam.py             # IAM-line from Hugging Face -> ../dataset/handwritten
python prepareenglishprinted.py   # Polish texts (Wolne Lektury) -> ../dataset/books
python createprintedpngs.py       # synthetic print -> ../dataset/printed (fonts in code/fonts/)
python mergeexcels.py             # combined ../dataset/labels.csv from all sources
python train/train.py --base 32 --ckpt-dir train/checkpoints_base32
```

Fonts (`code/fonts/`) are not included for licensing reasons. Copy any `.ttf` fonts with Polish characters into that folder.

### Fine-tuning on your own handwriting

1. Cut lines from photos of your notes:
   ```bash
   python imageoperations/inference.py --image note.jpg --checkpoint train/checkpoints_base32/best_crnn.pth --export ../dataset/my_handwriting
   ```
2. Correct the text in `../dataset/my_handwriting/labels.csv` (`text` column; an empty field skips the line). Mark a few pages as `validation`/`test`.
3. Merge the data and fine-tune the model:
   ```bash
   python mergeexcels.py
   python train/train.py --finetune train/checkpoints_base32/best_crnn.pth --base 32 --ckpt-dir train/checkpoints_finetune --lr 1e-4 --oversample my_handwriting=10 --epochs 40 --patience 10
   ```

IAM and print lines stay in training, so the model keeps working on other handwriting. Add more writers as separate folders in `dataset/`, so each gets its own CER in the logs.

### Limitations

- Handwritten lines that overlap through descenders and ascenders (g, y, ł) may be cut at the line boundary.
- Multi-column text is merged into one line.
- Curved lines near the notebook spine cannot be straightened with a single rotation angle.
- Very light pencil may be removed together with notebook ruling.
- Math formulas contain characters outside the alphabet (e.g. `=`, `+`).

### Structure

```
GUI/gui.py                     interface (tkinter) + PDF export
code/configs/train_config.yaml model alphabet
code/imageoperations/          preprocessing, segmentation, dataset, inference
code/model/crnn.py             CRNN network
code/train/                    training loop, metrics (CER, CTC decoding)
code/*.py                      data download and preparation
notebooks/colab_train.ipynb    training in Google Colab
photos/                        example results and training curves
```

---

## Polski

Program zamienia zdjęcie lub skan strony na tekst. Rozpoznaje **tekst drukowany oraz pismo odręczne (kursywę)** w języku **polskim i angielskim**. Alfabet modelu ma 93 znaki: litery polskie i angielskie, cyfry, interpunkcję i spację.

### Jak to działa

Potok ma dwa etapy: **wyodrębnienie linii tekstu** (OpenCV/NumPy) i **rozpoznanie linii** siecią neuronową.

#### 1. Wyodrębnianie tekstu (`code/imageoperations/preprocessing.py`)

1. **Binaryzacja:** rozmycie Gaussa i binaryzacja adaptacyjna (odporna na cień), potem usunięcie drobnego szumu.
2. **Prostowanie strony:** profil projekcji poziomej dla kątów ±5°. Wybierany jest kąt, przy którym linie tekstu są najbardziej poziome.
3. **Czyszczenie maski:** usunięcie linii zeszytu, marginesu, krawędzi kartki i cieni. Kryteria: jasność względem papieru i długie proste odcinki (operacje morfologiczne).
4. **Podział na linie:** każda linia tworzy maksimum w profilu poziomym. Granica jest w najgłębszym minimum między liniami, więc **wycinki nigdy na siebie nie nachodzą**.
5. **Normalizacja linii:** wyrównanie oświetlenia, usunięcie fragmentów sąsiednich linii, przycięcie do tuszu i skalowanie do wysokości 64 px. Ta sama funkcja działa w treningu i przy rozpoznawaniu.

#### 2. Rozpoznawanie (`code/model/crnn.py`)

Sieć **CRNN + CTC**, zbudowana od zera w PyTorch, bez pretrenowanych wag:
- **Część splotowa:** 6 bloków resztkowych, czyli 16 warstw splotowych z BatchNorm, LeakyReLU i Dropout2d. Rozmiar zmniejsza krok splotu, bez poolingu.
- **Część rekurencyjna:** 2-warstwowy dwukierunkowy LSTM (256 neuronów w każdym kierunku).
- **Wyjście:** warstwa liniowa na 94 klasy (93 znaki + blank CTC).
- **Rozmiar:** ok. 4,7 mln parametrów (wariant `--base 32`).

Sieć czyta całą linię naraz. Dzięki stracie CTC sama uczy się, gdzie stoi która litera, więc obrazu nie trzeba dzielić na pojedyncze znaki, co dla kursywy nie działa. Dekodowanie zachłanne CTC i metryka CER (odległość Levenshteina) są zaimplementowane samodzielnie (`code/train/metrics.py`).

### Dane i trening

**Zbiór uczący:**
- **IAM-line:** 10 373 linie angielskiego pisma odręcznego, oficjalny podział z rozłącznymi autorami w teście.
- **Syntetyczny druk:** 14 500 linii wygenerowanych z polskich i angielskich tekstów, 43 kroje pisma, zniekształcenia.
- **Własne pismo:** 243 ręcznie opisane linie polskich notatek (2 osoby), zdjęcia z telefonu.

W treningu działa augmentacja upodabniająca dane do zdjęć z telefonu: deformacje elastyczne, pochylenie, cień i szare tło, linie zeszytu, JPEG i niska rozdzielczość.

**Trening odbył się w Google Colab** (`notebooks/colab_train.ipynb`):

| Etap | Epoki | Val CER | Val dokładność linii |
|---|---|---|---|
| Trening bazowy (IAM + druk) | 34 | ≈ 2,7% | ≈ 66% |
| Dotrenowanie na własnym piśmie (`--finetune`, lr 1e-4, linie własne ×10) | 40 | ≈ 2,5% | ≈ 67% |

CER liczony jest na wspólnym zbiorze walidacyjnym (druk + IAM + własne pismo). Log treningu podaje też CER osobno dla każdego źródła. Przebieg treningu: [bazowy](photos/training_curves_epochs_40.png), [dotrenowanie](photos/training_curves_finetune.png).

### Uruchomienie

Wymagany Python 3.10+.

```bash
pip install -r code/requirements.txt
```

Na Windowsie z kartą NVIDIA zainstaluj PyTorcha z CUDA, np. `pip install torch --index-url https://download.pytorch.org/whl/cu128`. Do samego GUI wystarczy wersja na procesor.

**Wytrenowane modele (`*.pth`) nie są w repozytorium.** Umieść plik `best_crnn.pth` w `code/train/checkpoints_finetune/`. Jeśli go tam nie będzie, GUI przy starcie zapyta o plik.

#### GUI

```bash
python GUI/gui.py
```

Wybierz zdjęcie (`.png`/`.jpg`). Program zaznaczy wykryte linie, pokaże rozpoznany tekst do ręcznej poprawy i zapisze go jako notatkę PDF.

#### Z linii poleceń (z katalogu `code/`)

```bash
python imageoperations/inference.py --image strona.jpg --checkpoint train/checkpoints_finetune/best_crnn.pth --debug ramki.png
python imageoperations/inference.py --image linia.png --line      # obraz z jedną linią
```

### Przygotowanie danych i trening (z katalogu `code/`)

Folder `dataset/` nie jest w repozytorium. Tworzą go skrypty:

```bash
python downloadiam.py             # IAM-line z Hugging Face -> ../dataset/handwritten
python prepareenglishprinted.py   # teksty polskie (Wolne Lektury) -> ../dataset/books
python createprintedpngs.py       # syntetyczny druk -> ../dataset/printed (fonty w code/fonts/)
python mergeexcels.py             # wspólny ../dataset/labels.csv ze wszystkich źródeł
python train/train.py --base 32 --ckpt-dir train/checkpoints_base32
```

Fontów (`code/fonts/`) nie ma w repozytorium ze względu na licencję. Skopiuj do tego folderu dowolne fonty `.ttf` z polskimi znakami.

### Dotrenowanie na własnym piśmie

1. Wytnij linie ze zdjęć swoich notatek:
   ```bash
   python imageoperations/inference.py --image notatka.jpg --checkpoint train/checkpoints_base32/best_crnn.pth --export ../dataset/moje_pismo
   ```
2. Popraw tekst w `../dataset/moje_pismo/labels.csv` (kolumna `text`; puste pole oznacza pominięcie linii). Kilka stron oznacz jako `validation`/`test`.
3. Scal dane i dotrenuj model:
   ```bash
   python mergeexcels.py
   python train/train.py --finetune train/checkpoints_base32/best_crnn.pth --base 32 --ckpt-dir train/checkpoints_finetune --lr 1e-4 --oversample moje_pismo=10 --epochs 40 --patience 10
   ```

Linie IAM i druku nadal biorą udział w treningu, więc model nie traci skuteczności na innym piśmie. Kolejne osoby warto dodawać jako osobne foldery w `dataset/`, bo każdy dostaje wtedy własny CER w logach.

### Ograniczenia

- Linie pisma nachodzące na siebie wydłużeniami liter (g, y, ł) mogą zostać przycięte na granicy linii.
- Tekst w kilku kolumnach jest łączony w jedną linię.
- Wygięte linie przy grzbiecie zeszytu nie dają się wyprostować jednym kątem obrotu.
- Bardzo jasny ołówek może zostać usunięty razem z liniami zeszytu.
- Wzory matematyczne zawierają znaki spoza alfabetu (np. `=`, `+`).

### Struktura

```
GUI/gui.py                     interfejs (tkinter) + eksport PDF
code/configs/train_config.yaml alfabet modelu
code/imageoperations/          preprocessing, segmentacja, dataset, inferencja
code/model/crnn.py             sieć CRNN
code/train/                    pętla ucząca, metryki (CER, dekodowanie CTC)
code/*.py                      pobieranie i przygotowanie danych
notebooks/colab_train.ipynb    trening w Google Colab
photos/                        przykładowe wyniki i przebiegi treningu
```
