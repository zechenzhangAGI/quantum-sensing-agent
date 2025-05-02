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

def search_similar(query, embedding_file, top_k=3):
    """
    Search for similar contexts in the embedding file.
    
    Args:
        query: Query text
        embedding_file: Path to the embedding file
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with text, plot_references, and similarity score
    """
    try:
        # Load the stored embeddings
        text, stored_embeddings = load_embeddings(embedding_file)
        
        # Split the text into chunks (paragraphs or turns)
        chunks = text.split('\n')
        
        # Generate embeddings for each chunk
        chunk_embeddings = model.encode(chunks)
        
        # Generate embeddings for the query
        query_embedding = model.encode(query)
        
        # Create a FAISS index
        dimension = query_embedding.shape[0]
        index = faiss.IndexFlatL2(dimension)
        
        # Add the chunk embeddings to the index
        index.add(np.array(chunk_embeddings).astype('float32'))
        
        # Search for similar chunks
        distances, indices = index.search(np.array([query_embedding]).astype('float32'), min(top_k, len(chunks)))
        
        # Format the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):  # Ensure index is valid
                # Convert L2 distance to similarity score (higher is better)
                similarity = 1.0 / (1.0 + distances[0][i])
                
                # Get the chunk text and extract plot references
                chunk_text = chunks[idx]
                _, plot_refs = extract_plot_references(chunk_text)
                
                results.append({
                    "text": chunk_text,
                    "plot_references": plot_refs,
                    "score": float(similarity)
                })
        
        return results
    except Exception as e:
        print(f"Error in search_similar: {str(e)}")
        return []
