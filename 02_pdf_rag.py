import os
import numpy as np
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Client (International)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF and splits it into chunks (one chunk per page).
    """
    chunks = []
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return []

    print(f"Loading {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # We prepend "Page X: " so the AI knows where the info came from
                chunks.append(f"[Page {i+1}] {text}")
    
    print(f"Successfully loaded {len(chunks)} pages.")
    return chunks

def get_embedding(text):
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(
            model="text-embedding-v3", 
            input=[text],
            dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def find_best_match(query, corpus_embeddings, corpus_text):
    query_vec = get_embedding(query)
    if not query_vec: return None

    scores = []
    for doc_vec in corpus_embeddings:
        score = np.dot(query_vec, doc_vec) 
        scores.append(score)

    best_idx = np.argmax(scores)
    return corpus_text[best_idx]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load the PDF
    pdf_filename = "manual.pdf"  # <--- REPLACE THIS with your PDF filename
    kb_text = extract_text_from_pdf(pdf_filename)

    if not kb_text:
        print("No text found. Exiting.")
        exit()

    print("--- Indexing PDF Pages... ---")
    kb_vectors = [get_embedding(text) for text in kb_text]
    print("--- Indexing Complete! ---\n")

    while True:
        user_query = input("Ask a question about the PDF (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']: break

        # Retrieve
        best_context = find_best_match(user_query, kb_vectors, kb_text)
        
        # Generate
        prompt = f"""
        You are a helpful assistant. Use the PDF content below to answer.
        
        PDF Content: {best_context}
        
        Question: {user_query}
        """

        print("Thinking...")
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            print("\n--- Answer ---")
            print(completion.choices[0].message.content)
            print("--------------\n")
        except Exception as e:
            print(f"Error: {e}")
