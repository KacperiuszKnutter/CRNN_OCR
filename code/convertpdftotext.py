from pypdf import PdfReader
import os 

pdf_directory = "TO BE FIIILEED"  # Replace with the path to your PDF directory

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        reader = PdfReader(pdf_path)
        pages_text = [page.extract_text() for page in reader.pages[2:-1]]  # Skip the first two pages and the last page
        full_text = " ".join(pages_text)

        with open(os.path.join(pdf_directory, filename.replace(".pdf", ".txt")), "w", encoding="utf-8") as text_file:
            text_file.write(full_text)

print("PDF to text conversion completed!")