from transformers import AutoTokenizer, AutoModel
from pdf_processing import *
import torch

#Loads the BGE-M3 model
def load_bge_m3_model(model_name="BAAI/bge-m3"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

#Generates embedding for each chunk by using BGE-M3 model
def generate_bge_m3_embeddings(chunks, tokenizer, model):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the token embeddings as the sentence/paragraph embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return embeddings

def create_bge_m3_embeddings_for_pdf(url):
    #Process the PDF to get text chunks
    chunks = process_pdf(url)

    #Load the BGE-M3 model
    tokenizer, model = load_bge_m3_model()

    #Generate embeddings for each chunk using BGE-M3
    embeddings = generate_bge_m3_embeddings(chunks, tokenizer, model)

    return embeddings, chunks