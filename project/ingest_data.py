# project/ingest_data.py

import os
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------- CONFIG ----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)

UNIVERSITY_DOCS = os.path.join(ROOT_DIR, "data", "university_docs")
FEEDBACK_FILE = os.path.join(ROOT_DIR, "data", "student_feedback.csv")
VECTOR_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")

COLLECTION_NAME = "university_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- INGESTION ----------------
def load_documents():
    documents = []

    for file in os.listdir(UNIVERSITY_DOCS):
        path = os.path.join(UNIVERSITY_DOCS, file)

        if file.endswith(".pdf"):
            docs = PyPDFLoader(path).load()
            doc_type = "course_or_event"

        elif file.endswith(".txt"):
            docs = TextLoader(path).load()
            doc_type = "rules"

        elif file.endswith(".csv"):
            docs = CSVLoader(path).load()
            doc_type = "events"

        else:
            continue

        for d in docs:
            d.metadata.update({
                "source": file,
                "type": doc_type
            })

        documents.extend(docs)

    # Load student feedback
    feedback_docs = CSVLoader(FEEDBACK_FILE).load()
    for d in feedback_docs:
        d.page_content = f"Student Feedback: {d.page_content}"
        d.metadata.update({
            "source": "student_feedback",
            "type": "feedback"
        })

    documents.extend(feedback_docs)
    return documents


def build_vector_db():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    print("✅ Vector database built successfully")


if __name__ == "__main__":
    build_vector_db()
