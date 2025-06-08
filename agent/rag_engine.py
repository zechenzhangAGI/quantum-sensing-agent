import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """
    Generate embeddings for the given text.
    
    Args:
        text: Text to embed
        
    Returns:
        Numpy array of embeddings
    """
    return model.encode(text)

def save_embeddings(text, filepath):
    """
    Generate embeddings for text and save to a file.
    
    Args:
        text: Text to embed
        filepath: Path to save the embeddings
    """
    # Generate embeddings
    embeddings = embed_text(text)
    
    # Save embeddings and text
    data = {
        "text": text,
        "embeddings": embeddings.tolist()
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def load_embeddings(filepath):
    """
    Load embeddings from a file.
    
    Args:
        filepath: Path to the embeddings file
        
    Returns:
        Tuple of (text, embeddings)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data["text"]
    embeddings = np.array(data["embeddings"])
    
    return text, embeddings

def extract_plot_references(text):
    """
    Extract and highlight plot references in the text.
    
    Args:
        text: Text to process
        
    Returns:
        Tuple of (processed_text, plot_references)
    """
    plot_references = []
    lines = text.split('\n')
    
    for line in lines:
        if any(plot_type in line for plot_type in ['ESR_plot', 'FindNV_plot', 'GalvoScan_plot', 'Optimization_plot']):
            plot_references.append(line)
    
    return text, plot_references

def search_similar(query, embedding_file):
    """
    Search for similarity between a query and an entire text chunk in an embedding file.
    
    Args:
        query: Query text
        embedding_file: Path to the embedding file
        
    Returns:
        A dictionary with text, plot_references, and similarity score, or None.
    """
    try:
        # Load the stored text and its single embedding vector
        text, stored_embedding = load_embeddings(embedding_file)
        
        # Ensure stored_embedding is a 1D array
        stored_embedding = np.array(stored_embedding).flatten()

        # Generate embedding for the query
        query_embedding = model.encode(query).flatten()
        
        # Calculate Cosine Similarity
        dot_product = np.dot(query_embedding, stored_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_stored = np.linalg.norm(stored_embedding)
        
        if norm_query == 0 or norm_stored == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_query * norm_stored)

        # Get the chunk text and extract plot references
        _, plot_refs = extract_plot_references(text)
                
        return {
            "text": text,
            "plot_references": plot_refs,
            "score": float(similarity)
        }

    except Exception as e:
        print(f"Error in search_similar for file {embedding_file}: {str(e)}")
        return []
