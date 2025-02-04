import json
import re
import numpy as np
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def extract_experience_range(text: str):
    """
    Extracts the minimum and maximum years of experience from the given text.
    
    Returns a tuple (min_exp, max_exp) where:
      - For text like "3-5 years" or "3 - 5 years": returns (3, 5)
      - For text like "5+ years" or "5 years": returns (5, None)
      - If no number is found, returns (None, None)
    """
    # Look for a range indicated by a dash.
    range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', text)
    if range_match:
        min_exp = int(range_match.group(1))
        max_exp = int(range_match.group(2))
        return (min_exp, max_exp)
    # Otherwise, look for a single number.
    single_match = re.search(r'(\d+)', text)
    if single_match:
        min_exp = int(single_match.group(1))
        return (min_exp, None)
    return (None, None)

def aggregate_results(results, candidate_experience):
    """
    Aggregates search results by job_id while filtering based on experience.
    
    A job is accepted if:
      - The candidate's experience is not less than the job's minimum requirement, and
      - If a maximum is provided, the candidate's experience does not exceed it.
      
    For example, a candidate with 4 years (even if typed as "4+ years") qualifies for a job with "3-5 years"
    because 4 falls within [3, 5].
    
    Args:
        results: List of document objects (each with a `metadata` attribute and `page_content`).
        candidate_experience: Integer representing the candidate's experience.
        
    Returns:
        List of aggregated job info dictionaries.
    """
    aggregated_jobs = {}
    for doc in results:
        job_info = doc.metadata if doc.metadata else {}
        job_exp_text = job_info.get("experience_required", "")
        job_min, job_max = extract_experience_range(job_exp_text)
        
        if candidate_experience is not None and job_min is not None:
            if candidate_experience < job_min:
                continue
            if job_max is not None and candidate_experience > job_max:
                continue

        job_id = job_info.get("job_id")
        if not job_id:
            continue
        if job_id not in aggregated_jobs:
            aggregated_jobs[job_id] = job_info

    return list(aggregated_jobs.values())

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_jobs(query_text, experience_required, db_path, top_k=15, similarity_threshold=0.3):
    """
    Searches for relevant jobs using semantic search that takes into account:
      - The meaning of the user's query (via max marginal relevance search), and
      - The candidate's years of experience.
    
    A re‑ranking step is performed by computing the cosine similarity between the query
    embedding (as provided by the user) and each job's combined title and description embedding.
    Only jobs with a similarity above `similarity_threshold` are retained.
    
    For experience filtering, the candidate’s experience is simply extracted (e.g. "4+ years" is treated as 4)
    and a job is accepted if candidate_experience is within the job's range.
    
    Args:
        query_text (str): The user's query (e.g., "cloud ML").
        experience_required (str): The candidate's experience input (e.g., "4+ years").
        db_path (str): Path to the Chroma vector database.
        top_k (int): Number of top matches to initially retrieve.
        similarity_threshold (float): Minimum cosine similarity required (default is 0.3).
    
    Returns:
        str: JSON-formatted aggregated job search results.
    """
    # Use the robust embedding model: all-mpnet-base-v2 (768 dimensions)
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Load the Chroma vector database.
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
    
    # Retrieve an increased pool of candidates.
    results = db.max_marginal_relevance_search(query_text, k=top_k, fetch_k=top_k * 6)
    
    # Use the provided query (without manual augmentation) for embedding.
    query_embedding = embedding_function.embed_query(query_text)
    
    # Re-rank results using cosine similarity on the combined title and description.
    filtered_results = []
    for doc in results:
        combined_text = ""
        if doc.metadata and "title" in doc.metadata:
            combined_text += doc.metadata["title"] + " "
        combined_text += doc.page_content
        doc_embedding = embedding_function.embed_query(combined_text)
        sim = cosine_similarity(query_embedding, doc_embedding)
        if sim >= similarity_threshold:
            filtered_results.append(doc)
    
    # Extract candidate's experience as an integer.
    candidate_experience = None
    try:
        candidate_experience = int(re.search(r'\d+', experience_required).group())
    except Exception:
        pass

    aggregated_jobs = aggregate_results(filtered_results, candidate_experience)
    return json.dumps(aggregated_jobs, indent=4)
