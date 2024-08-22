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


#Indexes the embeddings in the Pinecone index with their metadata(chunk)
def index_embeddings(index, embeddings, metadata):
    print("uploading embeddings...")
    vectors = [(str(i), embedding, meta) for i, (embedding, meta) in enumerate(zip(embeddings, metadata))]
    index.upsert(vectors)


def setup_and_index_embeddings(api_key, environment, index_name, embeddings, chunks):
    print("setting up the index...")
    pc = initialize_pinecone(api_key)

    index = create_pinecone_index(pc, index_name, environment, dimension=len(embeddings) )

    metadata = [{'text_chunk': chunk} for chunk in chunks]

    index_embeddings(index, embeddings, metadata)

    return index

#Queries the Pinecone index
def query_pinecone_index(index, query_embedding, top_k=5):
    print("sending query to pinecone index...")

    #Ensure the query embedding is a list of floats
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    #Convert each element to float
    query_embedding = [float(x) for x in query_embedding]

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True,)
    return results


