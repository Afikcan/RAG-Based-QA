import requests
import PyPDF2
from io import BytesIO

#extracts text from the PDF file
def extract_text_from_pdf(url):
    print(f"Extracting text from {url}")
    response = requests.get(url)
    response.raise_for_status()

    pdf_file = BytesIO(response.content)

    #Read the PDF using PyPDF2
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

#Extracts text into chunks of 100 words.
def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:(i + chunk_size)]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_pdf(url):
    text = extract_text_from_pdf(url)
    chunks = split_text_into_chunks(text)
    return chunks
