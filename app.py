# Connect to Qdrant
from qdrant_client import QdrantClient
qdrant = QdrantClient(url="http://localhost:6333")

# Setup Groq
# Sign up on the Groq portal, generate an API key, and install the SDK.
pip install groq

# Setup Environment Variables
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Data Preparation and Processing
from datasets import load_dataset
# Load the fantasy sports dataset from Hugging Face
dataset = load_dataset('MicPie/unpredictable_baseball-fantasysports-yahoo-com')
# Display the first few entries to understand the data structure
print(dataset['train'].head())

# Preprocessing Data to Convert It into Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def preprocess_data(dataset, batch_size=32):
    text_data = [entry['text'] for entry in dataset['train']]
    embeddings_list = []

    # Process text data in batches
    for i in range(0, len(text_data), batch_size):
        batch = text_data[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings_list.append(batch_embeddings)

    # Concatenate all batch embeddings into a single numpy array
    embeddings = np.concatenate(embeddings_list, axis=0)
    return embeddings

# Generate embeddings for the dataset
embeddings = preprocess_data(dataset)

# Storing and Querying Data with Qdrant
from qdrant_client.http import models

def create_qdrant_collection(collection_name: str, vector_dim: int):
    try:
        qdrant_client.create_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        qdrant_client.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE))
        print(f"Collection '{collection_name}' created.")

create_qdrant_collection(collection_name='fantasy_sports', vector_dim=embeddings.shape[1])

def store_embeddings_in_qdrant(embeddings, metadata, collection_name='fantasy_sports'):
    points = []
    for i, embedding in enumerate(embeddings):
        point = models.PointStruct(id=i, vector=embedding.tolist(), payload=metadata[i])
        points.append(point)
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} embeddings in collection '{collection_name}'.")

# Preparing Metadata and Storing Embeddings
metadata = [{'text': entry['text']} for entry in dataset['train']]
store_embeddings_in_qdrant(embeddings, metadata)

# Querying Qdrant
def query_qdrant(query_embedding, collection_name='fantasy_sports', top_k=10):
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    return [hit.payload for hit in search_result]

query_text = "For a more aggressive team, which players should I choose?"
query_embedding = model.encode(query_text, convert_to_tensor=True).cpu().numpy()
result = query_qdrant(query_embedding)
print("Query Result:", result)

# Integrating with Groq for Interactive Queries
from langchain_groq import ChatGroq
chat_model = ChatGroq(model_name='llama3-8b-8192', api_key=groq_api_key, streaming=True)
from langchain_core.prompts import PromptTemplate
from langchain.memory.buffer import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
template = "You are an expert in sports games, including both real-world sports and fantasy sports. Your knowledge spans across various sports such as football, basketball, baseball, and more, with a strong focus on fantasy leagues and their intricacies. When answering queries, consider the following aspects, and use the provided variables {context} and {question} to tailor your response:"
prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

def generate_response(user_input: str) -> str:
    try:
        query_embedding = model.encode(user_input, convert_to_tensor=True).cpu().numpy()
        relevant_metadata = query_qdrant(query_embedding)
        context = " ".join([meta['text'] for meta in relevant_metadata])
        full_response = chat_model.predict(prompt_template.format(question=user_input, context=context))
        return full_response.strip()
    except Exception as e:
        print(f"An error occurred in generate_response: {str(e)}")
        return f"Error: {str(e)}"

# Building a User Interface with Streamlit
import streamlit as st

def app():
    st.title("Fantasy Sports Strategy AI")
    st.header("Step 1: Ask a Question")
    user_query = st.text_input("Ask a question about fantasy sports:", key='user_input')
    if st.button('Generate Response'):
        if user_query:
            response = generate_response(user_query)
            st.write(response)

if __name__ == "__main__":
    app()
