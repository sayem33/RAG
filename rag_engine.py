import openai
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding for text using OpenAI API."""
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def chunk_text(text, chunk_size=500):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_vector_store(pdf_path, pdf_content):
    """Create and store embeddings for PDF content."""
    file_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    store_path = os.path.join(VECTOR_STORE_DIR, f"{file_hash}.pkl")
    
    if os.path.exists(store_path):
        with open(store_path, 'rb') as f:
            return pickle.load(f)
    
    chunks = chunk_text(pdf_content)
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
    
    store_data = {"chunks": chunks, "embeddings": embeddings}
    with open(store_path, 'wb') as f:
        pickle.dump(store_data, f)
    
    return store_data

def retrieve_relevant_chunks(query, vector_store, top_k=3):
    """Retrieve top_k most relevant chunks for a query."""
    query_embedding = get_embedding(query)
    
    similarities = cosine_similarity(
        [query_embedding],
        vector_store["embeddings"]
    )[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [vector_store["chunks"][i] for i in top_indices]
    
    return "\n\n".join(relevant_chunks)

def rag_generate(prompt, pdf_path, pdf_content, model="gpt-4o"):
    """Generate content using RAG approach."""
    vector_store = create_vector_store(pdf_path, pdf_content)
    relevant_context = retrieve_relevant_chunks(prompt, vector_store)
    
    enhanced_prompt = f"Based on the following relevant content:\n\n{relevant_context}\n\n{prompt}"
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that generates content based on provided context."},
            {"role": "user", "content": enhanced_prompt}
        ],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

def rag_generate_quiz(pdf_path, pdf_content, difficulty):
    """Generate quiz using RAG approach."""
    vector_store = create_vector_store(pdf_path, pdf_content)
    
    query = f"Generate quiz questions about the main concepts at {difficulty} difficulty level"
    relevant_context = retrieve_relevant_chunks(query, vector_store, top_k=5)
    
    prompt = f"""
    You are a helpful teaching assistant. Based on the following course material:
    {relevant_context}

    Generate a quiz with the following requirements:
    - Difficulty Level: {difficulty}
    - Question types:
      - 'easy': MCQ (single correct answer) and True/False questions only.
      - 'medium': Include MCQ (multiple correct answers).
      - 'hard': Generate more complex variations of MCQs and True/False questions.
    - Provide correct answers for each question.
    - Format questions as JSON in this structure:
      [
        {{
            "question": "Sample question text",
            "type": "mcq_single / mcq_multiple / true_false",
            "options": ["Option1", "Option2", "Option3"],
            "answer": "Correct answer or list of correct answers"
        }}
      ]
    - Ensure the generated quiz is in valid JSON format.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

