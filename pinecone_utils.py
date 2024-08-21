import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np

#Initializes Pinecone
def initialize_pinecone(api_key):
    return Pinecone(api_key=api_key)


#Creates or connects to a Pinecone index
def create_pinecone_index(pc, index_name,environment, dimension):
    print("creating pinecone index...")
    if index_name not in pc.list_indexes().names():
        print("creating a new index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=environment
            )
        )
    index = pc.Index(index_name)
    print(pc.list_indexes())
    return index


#Indexes the embeddings in the Pinecone index
def index_embeddings(index, embeddings):
    print("uploading embeddings...")
    vectors = [(str(i), embedding) for i, embedding in enumerate(embeddings)]
    index.upsert(vectors)


def setup_and_index_embeddings(api_key, environment, index_name):
    print("setting up the index...")
    pc = initialize_pinecone(api_key)
    index = create_pinecone_index(pc, index_name, environment, dimension=1024 )
    #metadata = [{'text_chunk': chunk} for chunk in chunks]
    #index_embeddings(index, embeddings, metadata)
    return index

#Function to query the Pinecone index
def query_pinecone_index(index, query_embedding, top_k=5):
    print("sending query to pinecone index...")

    # Ensure the query embedding is a list of native Python floats
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # Convert each element to a native Python float if necessary
    query_embedding = [float(x) for x in query_embedding]

    results = index.query(vector=query_embedding, top_k=top_k)
    return results


