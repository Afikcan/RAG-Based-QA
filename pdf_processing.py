import requests
import PyPDF2
from io import BytesIO

#Downloads PDF from a URL
def download_pdf(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as file:  # Open a local file in write-binary mode.
        file.write(response.content)  # Write the downloaded content to the local file.
    print(f"PDF downloaded to {file_path}")  # Print confirmation of download.


#Extracts text from the downloaded PDF file
def extract_text_from_pdf(url):
    print(f"Extracting text from {url}")
    response = requests.get(url)
    response.raise_for_status()

    #Create a BytesIO object from the content
    pdf_file = BytesIO(response.content)

    #Read the PDF using PyPDF2
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

#Extracts text into chunks of approximately 100 words.
def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:(i + chunk_size)]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_pdf(url):
    text = extract_text_from_pdf(url)  # Extract text from the downloaded PDF.
    chunks = split_text_into_chunks(text)  # Split the extracted text into 100-word chunks.
    return chunks  # Return the chunks of text.
