import re
import nltk
import os

nltk.download("punkt")

def clean_and_tokenize(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(?:http|ftp)s?://\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    # Use simple whitespace-based tokenization instead of NLTK
    return text.split()

def format_questions(question):
    question = re.sub(r'\s+',' ',question).strip()
    return question

def format_document(doc):
    numbered_doc = '\n'.join([f"{i+1}.{os.path.basename(doc.metadata['source'])}:{doc.page_content}" for i, doc in enumerate(doc)])
    return numbered_doc