from dotenv import load_dotenv
import os
from embedding import create_bge_m3_embeddings_for_pdf, create_bge_m3_embeddings_for_query
from pinecone_utils import setup_and_index_embeddings, query_pinecone_index, index_embeddings
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

"""
uvicorn main:app --reload
"""

# Load environment variables from the .env file
load_dotenv()

# Get your credentials from environment variables
api_key = os.getenv('PINECONE_API_KEY')
environment = os.getenv('PINECONE_ENVIRONMENT')
index_name = 'glov-case-index'

# URL of the PDF you want to process
mistborn_secret_history_url = "https://www.freesad.com/media/books/PDF/2024-06-10/ea971c417c1442e2a7e2b174ab923509.pdf"
laws_of_soccer_url = "https://cdn1.sportngin.com/attachments/document/bb8a-2479018/FOSC_Laws_of_the_Game__simplified_.pdf"

#request body for API
class QueryRequest(BaseModel):
    url: str
    query: str

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
app = FastAPI()


# Initialize Pinecone index
index = setup_and_index_embeddings(api_key, environment, index_name)

print("I got started okay")

@app.get("/hi/")
def sayhi(text: str):
    return {"message": text}

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.post("/query/")
async def query_index(request: QueryRequest):
    # Step 1: Generate embeddings for the PDF from the provided URL
    embeddings, chunks = create_bge_m3_embeddings_for_pdf(request.url)

    #index = setup_and_index_embeddings(api_key, environment, index_name, embeddings, chunks)

    index_embeddings(index, embeddings)

    # Step 3: Generate the query embedding
    query_embedding, query_chunk = create_bge_m3_embeddings_for_query(request.query)

    print("my query:")
    print(query_embedding[0])
    print(query_chunk)

    if not query_embedding:
        raise HTTPException(status_code=400, detail="Query embedding could not be generated.")

    # Step 4: Query the Pinecone index
    results = query_pinecone_index(index, query_embedding[0], top_k=5)  # use the first embedding

    response = [{'id': res['id'], 'score': res['score']} for res in
                results['matches']]


    return response