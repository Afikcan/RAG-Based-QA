from dotenv import load_dotenv
import os
from embedding import create_bge_m3_embeddings_for_pdf
from pinecone_utils import setup_and_index_embeddings, query_pinecone_index
import numpy as np

# Load environment variables from the .env file
load_dotenv()

# Get your credentials from environment variables
api_key = os.getenv('PINECONE_API_KEY')
environment = os.getenv('PINECONE_ENVIRONMENT')
index_name = 'glov-case-index'

# URL of the PDF you want to process
mistborn_secret_history_url = "https://www.freesad.com/media/books/PDF/2024-06-10/ea971c417c1442e2a7e2b174ab923509.pdf"
laws_of_soccer_url = "https://cdn1.sportngin.com/attachments/document/bb8a-2479018/FOSC_Laws_of_the_Game__simplified_.pdf"

# Step 1: Generate embeddings for the PDF
embeddings, chunks = create_bge_m3_embeddings_for_pdf(laws_of_soccer_url)

# Step 2: Set up Pinecone and index the embeddings
index = setup_and_index_embeddings(api_key, environment, index_name, embeddings, chunks)

# Step 3: Query the index (for demonstration)
query_embedding = list(embeddings[0])  # Just an example query embedding

results = query_pinecone_index(index, query_embedding)
print(results)