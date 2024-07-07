import os
from openai import OpenAI
from config import LEARNING_OUTCOME_FILE
from pdf_processor import extract_text_from_pdf

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def generate_learning_outcomes(text):
    system_prompt = """
    You are an advanced AI that extracts learning outcomes and creates study guides for students.
    Your task is to analyze the text data from past exam questions and provide learning outcomes.
    List the learning outcomes as bullet points and provide a detailed explanation for each one.
    Aim to find at least 20 learning outcomes from the text.
    """

    user_prompt = f"Extract the learning outcomes from the following text:\n{text}\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def refine_outcomes(outcomes):
    system_prompt = """
    You are an AI that refines and consolidates learning outcomes.
    Given a list of learning outcomes from different sources, combine them, eliminate duplicates, and organize them by importance.
    Provide detailed explanations for each outcome.
    """

    user_prompt = f"Refine and consolidate the following learning outcomes:\n{outcomes}\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def process_pdfs_for_outcomes(pdf_folder):
    outcomes = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text_content = extract_text_from_pdf(pdf_path)
            pdf_outcomes = generate_learning_outcomes(text_content)
            outcomes.append(pdf_outcomes)

    combined_outcomes = "\n\n".join(outcomes)
    refined_outcomes = refine_outcomes(combined_outcomes)

    with open(LEARNING_OUTCOME_FILE, 'w', encoding='utf-8') as file:
        file.write(refined_outcomes)

    return refined_outcomes
