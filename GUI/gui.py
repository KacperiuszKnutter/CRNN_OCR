"""
Prosty interfejs OCR: wybierz zdjęcie/skan (.png/.jpg) -> odczytany tekst -> notatka PDF.
Uruchamianie z folderu projektu:  python GUI/gui.py
Model: code/train/checkpoints_base32/best_crnn.pth (jeśli go nie ma, okno zapyta o plik .pth).
"""
import io
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Button, Label, Frame, Text, filedialog, messagebox, END, BOTH, LEFT, RIGHT, X

import cv2
import torch
from PIL import Image, ImageTk

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from imageoperations.dataset import load_vocab
from imageoperations.inference import read_page
from imageoperations.preprocessing import load_gray
from model.crnn import CRNN

# for testing
DEFAULT_CHECKPOINT = CODE_DIR / "train" / "checkpoints_finetune" / "best_crnn.pth"
#DEFAULT_CHECKPOINT = CODE_DIR / "train" / "checkpoints_base32" / "best_crnn.pth"
WINDOW_W, WINDOW_H = 800, 600


def load_model(checkpoint, device):
    _, idx_to_char, num_classes = load_vocab(CODE_DIR / "configs" / "train_config.yaml")
    ckpt = torch.load(checkpoint, map_location=device)
    model = CRNN(num_chars=num_classes, base=ckpt.get("base", 16)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, idx_to_char


def save_pdf(path, text, source_name, image=None):
    # PDF z polskimi znakami: font DejaVu Sans dołączony do matplotliba
    from matplotlib import font_manager
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as PdfImage, Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    pdfmetrics.registerFont(TTFont("DejaVu", font_manager.findfont("DejaVu Sans")))
    title = ParagraphStyle("title", fontName="DejaVu", fontSize=16, leading=20)
    small = ParagraphStyle("small", fontName="DejaVu", fontSize=9, leading=12, textColor="#666666")
    body = ParagraphStyle("body", fontName="DejaVu", fontSize=12, leading=17)

    story = [Paragraph("Notatka", title), Spacer(1, 4),
             Paragraph(escape(f"Źródło: {source_name} · {datetime.now():%d.%m.%Y %H:%M}"), small), Spacer(1, 14)]
    story += [Paragraph(escape(line) or "&nbsp;", body) for line in text.splitlines()]
    if image is not None:  # oryginał z zaznaczonymi liniami na końcu notatki
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        scale = min(16 * cm / image.width, 22 * cm / image.height)  # zmieść na stronie A4
        story += [Spacer(1, 20), PdfImage(buf, width=image.width * scale, height=image.height * scale)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)


class OcrApp:
    def __init__(self, root, model, idx_to_char, device):
        self.root, self.model, self.idx_to_char, self.device = root, model, idx_to_char, device
        self.image_path, self.preview_image = None, None
        root.title("OCR - obraz na tekst")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")  # zawsze ten sam rozmiar przy starcie
        root.minsize(WINDOW_W, WINDOW_H)

        bar = Frame(root)
        bar.pack(fill=X, padx=8, pady=6)
        Button(bar, text="Wybierz obraz…", command=self.choose_image).pack(side=LEFT)
        self.pdf_button = Button(bar, text="Zapisz notatkę PDF…", command=self.export_pdf, state="disabled")
        self.pdf_button.pack(side=LEFT, padx=6)
        self.status = Label(bar, text=f"Model wczytany ({device}). Wybierz plik .png lub .jpg.", anchor="w")
        self.status.pack(side=LEFT, padx=10)

        body = Frame(root)
        body.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        self.preview = Label(body, bg="#dddddd", width=1)
        self.preview.pack(side=LEFT, fill=BOTH, expand=True)
        # tekst można poprawić ręcznie przed zapisem PDF
        self.text = Text(body, wrap="word", font=("Segoe UI", 11), width=1)
        self.text.pack(side=RIGHT, fill=BOTH, expand=True, padx=(8, 0))

    def choose_image(self):
        path = filedialog.askopenfilename(title="Wybierz zdjęcie lub skan",
                                          filetypes=[("Obrazy", "*.png *.jpg *.jpeg"), ("Wszystkie pliki", "*.*")])
        if not path:
            return
        self.image_path = Path(path)
        self.status.config(text=f"Rozpoznawanie: {self.image_path.name}…")
        self.pdf_button.config(state="disabled")
        # OCR w osobnym wątku, żeby okno nie zamarzło
        threading.Thread(target=self.run_ocr, daemon=True).start()

    def run_ocr(self):
        try:
            text, gray, boxes = read_page(self.model, load_gray(self.image_path), self.idx_to_char, self.device)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            for x, y, w, h in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.root.after(0, self.show_result, text, Image.fromarray(vis), len(boxes))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Błąd", str(e)))

    def show_result(self, text, image, n_boxes):
        self.preview_image = image
        thumb = image.copy()
        thumb.thumbnail((max(self.preview.winfo_width(), 200), max(self.preview.winfo_height(), 200)))
        self.preview.photo = ImageTk.PhotoImage(thumb)  # referencja, inaczej Tk zgubi obraz
        self.preview.config(image=self.preview.photo)
        self.text.delete("1.0", END)
        self.text.insert("1.0", text)
        self.pdf_button.config(state="normal")
        self.status.config(text=f"{self.image_path.name}: {n_boxes} fragmentów tekstu. Możesz poprawić tekst przed zapisem.")

    def export_pdf(self):
        path = filedialog.asksaveasfilename(title="Zapisz notatkę", defaultextension=".pdf",
                                            initialfile=f"{self.image_path.stem}_notatka.pdf",
                                            filetypes=[("PDF", "*.pdf")])
        if path:
            save_pdf(path, self.text.get("1.0", END).rstrip(), self.image_path.name, self.preview_image)
            self.status.config(text=f"Zapisano: {path}")


def main():
    root = Tk()
    checkpoint = DEFAULT_CHECKPOINT
    if not checkpoint.exists():
        chosen = filedialog.askopenfilename(title="Wskaż wytrenowany model (best_crnn.pth)",
                                            filetypes=[("Model PyTorch", "*.pth *.pt")])
        if not chosen:
            return
        checkpoint = Path(chosen)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_char = load_model(checkpoint, device)
    OcrApp(root, model, idx_to_char, device)
    root.mainloop()


if __name__ == "__main__":
    main()
