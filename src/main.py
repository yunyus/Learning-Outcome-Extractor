import os
from pdf_processor import load_or_process_pdfs
from vector_store import load_or_store_vectors
from similarity_search import assign_points_to_pages, save_relevant_pages
from outcome_generator import process_pdfs_for_outcomes
from config import CHECKPOINT_FOLDER, LEARNING_OUTCOME_FILE, PAST_EXAM_QUESTIONS_FILE


def main():
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    vector_db = load_or_process_pdfs()
    index, model = load_or_store_vectors(vector_db)

    if not os.path.exists(LEARNING_OUTCOME_FILE):
        queries = process_pdfs_for_outcomes(PAST_EXAM_QUESTIONS_FILE)
    else:
        with open(LEARNING_OUTCOME_FILE, 'r', encoding='utf-8') as file:
            queries = file.readlines()

    points_db = assign_points_to_pages(queries, index, model, vector_db)
    for (pdf_name, page_num), points in points_db.items():
        print(f"PDF: {pdf_name}, Page: {page_num}, Points: {points}")

        with open("points.txt", 'a') as f:
            f.write(f"PDF: {pdf_name}, Page: {page_num}, Points: {points}\n")

    save_relevant_pages(queries, index, model, vector_db, "relevant_pages.txt")


if __name__ == "__main__":
    main()
