from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_job_descriptions(jobs_df):
    """
    Process job descriptions from the DataFrame into document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    documents = []
    for _, row in jobs_df.iterrows():
        job_desc = str(row['Job Description'])
        job_id = str(row['Job ID'])
        metadata = {
            "job_id": job_id,
            "title": row["Job Title"],
            "company": row["Company"],
            "location": row["Location"],
            "experience_required": row["Experience Required"]
        }
        chunks = text_splitter.create_documents([job_desc], metadatas=[metadata])
        documents.extend(chunks)
    return documents
