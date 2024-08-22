from dotenv import load_dotenv
import os
from embedding import create_bge_m3_embeddings_for_pdf, create_bge_m3_embeddings_for_query
from pinecone_utils import setup_and_index_embeddings, query_pinecone_index, index_embeddings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mangum import Mangum

"""
uvicorn main:app --reload
"""

# Load environment variables from the .env file
load_dotenv()

# Get your credentials from environment variables
api_key = os.getenv('PINECONE_API_KEY')
environment = os.getenv('PINECONE_ENVIRONMENT')
index_name = 'glov-case-index'

#request body for API
class QueryRequest(BaseModel):
    url: str
    query: str

app = FastAPI()

handler = Mangum(app)

# Initialize Pinecone index
index = None

@app.get("/")
def sayhi():
    return {"Hello": "Glov"}

@app.post("/query/")
async def query_index(request: QueryRequest):
    global index

    # Step 1: Generate embeddings for the PDF from the provided URL
    embeddings, chunks = create_bge_m3_embeddings_for_pdf(request.url)

    if index is None:
        index = setup_and_index_embeddings(api_key, environment, index_name, embeddings, chunks)
    else:
        index_embeddings(index, embeddings, [{'text_chunk': chunk} for chunk in chunks])

    # Step 3: Generate the query embedding
    query_embedding, query_chunk = create_bge_m3_embeddings_for_query(request.query)

    if not query_embedding:
        raise HTTPException(status_code=400, detail="Query embedding could not be generated.")

    # Step 4: Query the Pinecone index
    results = query_pinecone_index(index, query_embedding[0], top_k=5)  # use the first embedding

    response = [{'id': res['id'], 'score': res['score'], 'text_chunk': res['metadata']['text_chunk']} for res in results['matches']]

    return response