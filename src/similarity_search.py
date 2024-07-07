import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import TOP_K


def perform_similarity_search(query, index, model, vector_db):
    query_vector = model.encode([query])
    _, top_indices = index.search(
        np.array(query_vector, dtype=np.float32), TOP_K)
    top_indices = top_indices[0]

    top_pages = [(vector_db[i][0], vector_db[i][1], vector_db[i][2])
                 for i in top_indices]
    return top_pages


def assign_points_to_pages(queries, index, model, vector_db):
    points_db = {}
    for query in queries:
        top_pages = perform_similarity_search(query, index, model, vector_db)
        for i, (pdf_name, page_num, _) in enumerate(top_pages):
            if (pdf_name, page_num) not in points_db:
                points_db[(pdf_name, page_num)] = 0
            points_db[(pdf_name, page_num)] += TOP_K - i
    return points_db


def save_relevant_pages(queries, index, model, vector_db, output_file):
    relevant_pages = {}
    for query in queries:
        top_pages = perform_similarity_search(query, index, model, vector_db)
        relevant_pages[query] = [
            f"Page {page_num}" for _, page_num, _ in top_pages]

    with open(output_file, 'w') as f:
        for query, pages in relevant_pages.items():
            f.write(f"{query} = {', '.join(pages)}\n")
