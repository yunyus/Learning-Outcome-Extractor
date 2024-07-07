import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from utils import save_checkpoint, load_checkpoint
from config import MODEL_NAME, CHECKPOINT_FOLDER


def store_vectors(vector_db):
    model = SentenceTransformer(MODEL_NAME)
    texts = [text for _, _, text in vector_db]
    vectors = model.encode(texts)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))

    return index, model


def load_or_store_vectors(vector_db):
    index_checkpoint = os.path.join(CHECKPOINT_FOLDER, "index.faiss")
    model_checkpoint = os.path.join(CHECKPOINT_FOLDER, "model.pkl")
    if os.path.exists(index_checkpoint) and os.path.exists(model_checkpoint):
        index = faiss.read_index(index_checkpoint)
        model = load_checkpoint(model_checkpoint)
    else:
        index, model = store_vectors(vector_db)
        faiss.write_index(index, index_checkpoint)
        save_checkpoint(model, model_checkpoint)

    return index, model
