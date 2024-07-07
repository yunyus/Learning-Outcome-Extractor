import torch

# Configuration settings
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
PDF_FOLDER = "./data/pdfs/"
CHECKPOINT_FOLDER = "./data/checkpoints/"
LEARNING_OUTCOME_FILE = "./data/learning_outcomes.txt"
PAST_EXAM_QUESTIONS_FILE = "./data/past_exams/"
MAX_TEXT_LENGTH = 512
TOP_K = 3

# Use GPU if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
