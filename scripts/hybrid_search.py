import os, ast
import json
import re
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration - Get API key from https://console.groq.com
GROQ_API_KEY = "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"  # Set in environment variables
client = Groq(api_key=GROQ_API_KEY)

def extract_experience_range(text: str):
    range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', text)
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))
    single_match = re.search(r'(\d+)', text)
    return (int(single_match.group(1)), None) if single_match else (None, None)

def aggregate_results(results, candidate_exp):
    jobs = {}
    for doc in results:
        meta = doc.metadata or {}
        exp_text = meta.get("experience_required", "")
        min_exp, max_exp = extract_experience_range(exp_text)
        
        if candidate_exp and min_exp:
            if candidate_exp < min_exp or (max_exp and candidate_exp > max_exp):
                continue
                
        if job_id := meta.get("job_id"):
            jobs[job_id] = {
                "job_id": job_id,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "experience_required": exp_text
            }
    return list(jobs.values())

def query_groq_llama3(context, query, experience):
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
    "role": "system",
    "content": """You are an expert job matching AI. Analyze listings using these rules:
    
    1. Semantic Understanding: Interpret the query's core meaning and related terms
       - For example -"cloud ML" = (cloud computing + machine learning) | AWS/Azure/GCP + ML/AI/Data Science + DevOps 
       - Include roles with either/both domains but exclude unrelated ones
    
    2. Experience Validation:
       - Strictly match experience requirements using this logic:
         if user_exp < job_min: reject
         if job_max exists and user_exp > job_max: reject
    
    3. Relevance Filtering:
       - Keep only roles combining cloud+ML or individual domains
       - Reject roles with <50% relevance to query's semantic core

    4. Always take job description into consideration while filtering. Thats the most important part while doing the filter.
    Understand the meaning and intent behind the job description, analyze carefully and then return the jobs.
    
    Return JSON format (ALWAYS ESCAPE QUOTES):
    {"matches": [{
        "job_id": "id",
        "title": "Job Title",
        "company": "Company",
        "location": "Location",
        "experience_required": "X-Y years",
        "match_reason": "Cloud infrastructure + ML pipelines"
    }]
    ALWAYS return valid JSON with DOUBLE QUOTES using this format:
                {json.dumps({"matches": [{
                    "job_id": "example_id",
                    "title": "Example Title",
                    "company": "Example Company",
                    "experience_required": "X-Y years"
                }]})}
                
                RULES:
                1. Use " for ALL JSON syntax (never ')
                2. Escape internal quotes with \\
                3. Validate JSON before returning}"""
}, {
    "role": "user",
    "content": f"""QUERY ANALYSIS REQUEST
    Search Context: {context}
    ────────────────────────────────
    User Query: {query}
    User Experience: {experience}
    ────────────────────────────────
    Analyze job matches using:
    - Semantic keyword relationships
    - Experience range validation
    - Domain relevance scoring"""
}]
,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1024
        )
        print(json.loads(response.choices[0].message.content)["matches"])
        # Validate and parse response
        if response.choices:
            raw_content = response.choices[0].message.content
            
            # Fix common JSON issues
            json_str = raw_content.replace("'", '"')  # Replace single with double quotes
            json_str = re.sub(r'(?<!\\)"', '\\"', json_str)  # Escape unescaped internal quotes
            json_str = json_str.replace('\\\"', '\\\\"')  # Handle existing escapes
            
            try:
                parsed = json.loads(json_str)
                return parsed.get("matches", [])
            except json.JSONDecodeError:
                print("JSON parsing failed. Attempting repair...")
                # Try parsing with literal evaluation
                try:
                    parsed = ast.literal_eval(raw_content)
                    return json.loads(json.dumps(parsed).replace("'", '"'))
                except:
                    print("Final parsing failed. Raw response:", raw_content)
                    return []
                
        return []
        
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {str(e)}\nRaw content: {raw_content}")
        return []
    except Exception as e:
        print(f"API Error: {str(e)}")
        return []



def hybrid_search_jobs(query_text, experience_required, db_path, top_k=15):
    # Initialize Chroma DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Retrieve documents
    results = db.max_marginal_relevance_search(query_text, k=top_k)
    
    # Build context
    context = "\n---\n".join(
        f"ID: {doc.metadata.get('job_id', '')}\n"
        f"Title: {doc.metadata.get('title', '')}\n"
        f"Company: {doc.metadata.get('company', '')}\n"
        f"Experience: {doc.metadata.get('experience_required', '')}\n"
        f"Location: {doc.metadata.get('location', '')}\n"
        f"Description: {doc.page_content[:500]}..."
        for doc in results if doc.metadata
    )
    
    # Get LLM response
    response = query_groq_llama3(context, query_text, experience_required)
    return response["matches"]


