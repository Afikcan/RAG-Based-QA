RAG-Based Question-Answering System Documentation
Overview

This project is a RESTful API service designed to extract information from PDF documents and answer questions based on the content. The API uses a RAG (Retrieval-Augmented Generation) approach, leveraging a transformer model for generating embeddings, which are indexed and queried through a Pinecone vector database.
Requirements

To run this project, ensure that you have the following Python packages installed, as specified in the requirements.txt file:

    numpy==1.26.4
    pinecone==5.0.1
    PyPDF2==2.10.5
    python-dotenv==0.21.0
    Requests==2.32.3
    torch==2.3.0
    transformers==4.44.1
    fastapi==0.103.0
    uvicorn==0.30.6
GPT

Let's create comprehensive documentation for your project, which will guide users on how to understand and use the code. Here's an outline for the documentation, followed by detailed sections:
RAG-Based Question-Answering System Documentation
Overview

This project is a RESTful API service designed to extract information from PDF documents and answer questions based on the content. The API uses a RAG (Retrieval-Augmented Generation) approach, leveraging a transformer model for generating embeddings, which are indexed and queried through a Pinecone vector database.
Requirements

To run this project, ensure that you have the following Python packages installed, as specified in the requirements.txt file:

    numpy==1.26.4
    pinecone==5.0.1
    PyPDF2==2.10.5
    python-dotenv==0.21.0
    Requests==2.32.3
    torch==2.3.0
    transformers==4.44.1
    fastapi==0.103.0
    mangum==0.17.0
    uvicorn==0.30.6

To install the dependencies, run:

bash

pip install -r requirements.txt

Project Structure

The project consists of the following key Python files:

    pdf_processing.py: Handles downloading the PDF from a URL, extracting text, and splitting the text into manageable chunks.
    embedding.py: Manages the generation of text embeddings using the BGE-M3 model from Hugging Face.
    pinecone_utils.py: Provides utilities to initialize and interact with a Pinecone vector database, including indexing and querying embeddings.
    main.py: Implements the FastAPI service, defining endpoints to process a PDF, generate embeddings, index them in Pinecone, and handle queries.

Functionality
1. PDF Processing (pdf_processing.py)

    extract_text_from_pdf(url): Downloads and extracts text from a PDF at the specified URL.
    split_text_into_chunks(text, chunk_size=100): Splits the extracted text into chunks of a specified word length (default: 100 words).

2. Embedding Generation (embedding.py)

    load_bge_m3_model(model_name="BAAI/bge-m3"): Loads the BGE-M3 model and tokenizer from Hugging Face.
    generate_bge_m3_embeddings(chunks, tokenizer, model): Generates embeddings for each text chunk using the loaded BGE-M3 model.
    create_bge_m3_embeddings_for_pdf(url): Combines PDF processing and embedding generation to create embeddings for a given PDF.
    create_bge_m3_embeddings_for_query(query): Generates embeddings for a given query string.

3. Pinecone Utilities (pinecone_utils.py)

    initialize_pinecone(api_key): Initializes the Pinecone service with the provided API key.
    create_pinecone_index(pc, index_name, environment, dimension): Creates a Pinecone index if it doesn’t already exist.
    index_embeddings(index, embeddings, metadata): Indexes the provided embeddings along with their corresponding metadata.
    setup_and_index_embeddings(api_key, environment, index_name, embeddings, chunks): Sets up the Pinecone index and indexes the embeddings.
    query_pinecone_index(index, query_embedding, top_k=5): Queries the Pinecone index for the top k results matching the query embedding.

4. API Implementation (main.py)

    Endpoint / (GET): A simple test endpoint returning a greeting.
    Endpoint /query/ (POST): Accepts a URL and query string, processes the PDF at the URL, generates and indexes embeddings, and returns the most relevant text chunks based on the query.
   
Example Request:

json

{
  "url": "https://cdn1.sportngin.com/attachments/document/bb8a-2479018/FOSC_Laws_of_the_Game__simplified_.pdf",
  "query": "Under the simplified laws, what conditions must be met for a referee to issue a yellow card to a player during a soccer match?"
}

Example Response:

json

[
    {
        "id": "13",
        "score": 0.828873813,
        "text_chunk": "any other unmentioned offense Yellow cards are awarded as a caution or warning to a player and can be issued for the following offenses: /square4 Unsporting behavior /square4 Dissent by word or action /square4 Persistent infringement of the Laws of the Game /square4 Delaying the restart of play /square4 Failure to respect the required distance when play is restarted with a corner kick, free kick, or throw-in /square4 Entering or re-entering the field of play without t he referee’s permission /square4 deliberately leaving the field of play without the referee’s permission Red cards are used to send a player off"
    },
    ...
]

Deployment

The API can be deployed using Docker container and AWS ECS. Here's a simple deployment guide:

Running Locally

To run the API locally:

bash
uvicorn main:app --reload

This will start the API on http://127.0.0.1:8000.

Docker Deployment

    Build Docker Image:

    bash

docker build -t rag-api .

Then you will be able to run your docker container on your local. 

AWS Deployment

Refer to the AWS ECS tutorial linked in the references for detailed steps on deploying the API on AWS.

AWS Deployment Guide
Step 1: Push Docker Image to Docker Hub

After building your Docker image locally, the first step is to push this image to Docker Hub. This allows your image to be accessible from AWS ECS (Elastic Container Service).

Commands:

bash

docker login
docker tag local-image:tagname afique/rag-based-qa:latest
docker push afique/rag-based-qa:latest

Step 2: Create Task Definitions in AWS ECS

A Task Definition in AWS ECS is like a blueprint for your application. It defines which Docker image to use, the necessary CPU and memory resources, and the port mappings. By specifying the containerPort and hostPort as 8000, you ensure that your API will be accessible on port 8000 once deployed.

Why create a Task Definition?

    It encapsulates the configuration needed to run containers in AWS ECS.
    It allows you to manage and version the deployment configuration, enabling easy updates and rollbacks.

Example Task Definition:

json

{
    "family": "glov-case",
    "containerDefinitions": [
        {
            "name": "glov-case-container",
            "image": "afique/rag-based-qa:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "hostPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/glov-case",
                    "awslogs-region": "eu-west-3",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ],
    "requiresCompatibilities": ["FARGATE"],
    "networkMode": "awsvpc",
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::156041399827:role/ecsTaskExecutionRole"
}

Step 3: Create an ECS Cluster

An ECS Cluster is a logical grouping of tasks or services. When you create a service within this cluster, AWS manages the deployment and scaling of your containers based on the task definitions.

Why create a Cluster?

    It provides a management layer for running your containerized applications across multiple EC2 instances or using Fargate.

Step 4: Create a Service in the ECS Cluster

Next, create a service in your ECS cluster using the task definition. This service ensures that the specified number of instances of your task are running at all times. It also manages the deployment, allowing for zero-downtime updates.

Why create a Service?

    It ensures high availability and reliability by maintaining the desired number of task instances.
    It provides load balancing and auto-scaling capabilities.

Step 5: Adjust Security Group Inbound Rules

To allow external access to your API, you need to modify the inbound rules of the security group associated with your ECS service. Specifically, you should allow incoming traffic on port 8000.

Why adjust Security Group Inbound Rules?

    To enable your API to receive requests from outside the AWS environment, making it accessible over the internet.

Step 6: Access the API

Once the deployment is complete, and the security group rules are updated, you can access your API at the public IP address assigned to your service on the specified port. For example:

url

http://15.188.195.103:8000

You can now send requests to this URL to interact with your RAG-based Question-Answering system.
