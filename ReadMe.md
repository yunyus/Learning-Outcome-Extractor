# Learning Outcome Extractor

This project provides a framework for extracting learning outcomes from PDFs, generating study guides using GPT, and storing and searching relevant passages using FAISS. The project includes data loading, preprocessing, model training, and inference functionalities.

## Project Structure

```
learning_outcome_extractor/
│
├── data/
│   └── pdfs/
│       ├── example1.pdf
│       ├── example2.pdf
│   └── checkpoints/
│       ├── index.faiss
│       ├── model.pkl
│       ├── vector_db.pkl
│   └── learning_outcomes.txt
├── src/
│   ├── config.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   ├── similarity_search.py
│   ├── outcome_generator.py
│   ├── main.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Place your PDF files in the `data/pdfs` directory.

4. Place the past exam questions in the `data/past_exams` directory.

## Usage

To process PDFs, generate learning outcomes, and find the slide pages that is correleated with the past exam questions, run:

```
python src/main.py
```
