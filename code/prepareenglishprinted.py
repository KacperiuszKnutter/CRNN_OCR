# from datasets import load_dataset


# wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
# english_text = "".join(row["text"] for row in wiki if len(row["text"]) > 50)

# with open("../dataset/books/eng/english.txt", "w", encoding="utf-8") as f:
#     f.write(english_text)

# for polish text down below ^^^ uncomment for use

import urllib.request

books = {
    "makbet.txt": "https://wolnelektury.pl/media/book/txt/makbet.txt",
    "lalka_t1.txt": "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt",
    "quo_vadis.txt": "https://wolnelektury.pl/media/book/txt/quo-vadis.txt",
}

for filename, url in books.items():
    urllib.request.urlretrieve(url, f"../dataset/books/pl/{filename}")
    print(f"Downloaded {filename} from {url}")