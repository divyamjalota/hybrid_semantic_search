import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_chroma_db(documents, db_path):
    """
    Create and populate the Chroma vector database with the job documents.
    
    If the database directory exists, it will be removed first.
    """
    if os.path.exists(db_path):
        print("Clearing existing Chroma database...")
        shutil.rmtree(db_path)  # Cross‑platform removal

    # Use a powerful embedding model.
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

    # (Optional) Print a test embedding.
    print("Test query embedding:", embedding_function.embed_query("Test query"))

    db = Chroma(embedding_function=embedding_function, persist_directory=db_path)
    db.add_documents(documents)
    #db.persist()
    print("Chroma database populated successfully.")
