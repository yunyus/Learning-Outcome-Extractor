import os
import fitz  # PyMuPDF
import pickle
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint
from config import PDF_FOLDER, CHECKPOINT_FOLDER


def extract_text_from_pdf(pdf_path):
    text_content = ""
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text_content += f"<Page {page_num + 1}>\n{page.get_text()}\n\n"

    return text_content


def process_pdfs():
    vector_db = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            vector_db.append((pdf_file, page_num, page_text))

        checkpoint_path = os.path.join(
            CHECKPOINT_FOLDER, f"{pdf_file}_checkpoint.pkl")
        save_checkpoint(vector_db, checkpoint_path)

    return vector_db


def load_or_process_pdfs():
    vector_db_checkpoint = os.path.join(CHECKPOINT_FOLDER, "vector_db.pkl")
    if os.path.exists(vector_db_checkpoint):
        return load_checkpoint(vector_db_checkpoint)
    else:
        vector_db = process_pdfs()
        save_checkpoint(vector_db, vector_db_checkpoint)
        return vector_db
