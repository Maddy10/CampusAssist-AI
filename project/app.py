# project/app.py

import streamlit as st
import os
import pandas as pd

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

# ---------------- CONFIG ----------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)

VECTOR_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
FEEDBACK_FILE = os.path.join(ROOT_DIR, "data", "student_feedback.csv")

LLM_MODEL = "phi3:mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "university_knowledge"

st.set_page_config("UniSupport AI", "🎓", layout="wide")
st.title("🎓 University Student Support AI")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL, temperature=0.3)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

llm = load_llm()
db = load_vector_db()

# ---------------- RAG TOOL ----------------
rag_prompt = PromptTemplate(
    template="""
You are a question-answering assistant that must rely strictly on the provided context.

Instructions:
- Answer the question using ONLY the information explicitly stated in the context.
- Do NOT use prior knowledge or make assumptions.
- If the answer cannot be found verbatim or clearly inferred from the context, respond exactly with:
  "I don't know."

Context:
{context}

Question:
{question}

Answer (based only on the context):
""",
    input_variables=["context", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": rag_prompt}
)

# ---------------- FEEDBACK SUMMARIZER ----------------
feedback_prompt = PromptTemplate(
    template="""
You are an academic feedback analyst.

Analyze the student feedback provided below and produce a concise, well-organized summary that includes:
- Overall sentiment (positive, neutral, or negative)
- Key positive themes mentioned by students
- Most frequent complaints or pain points
- Notable patterns or recurring issues

Student Feedback:
{feedback}

Feedback Summary:
""",
    input_variables=["feedback"]
)


feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt)

# ---------------- REPORT GENERATOR ----------------
report_prompt = PromptTemplate(
    template="""
You are an educational analyst tasked with writing a professional student satisfaction report.

Using the feedback summary provided below, generate a well-structured report that includes:
- An overall satisfaction overview
- Key strengths highlighted by students
- Common concerns or areas for improvement
- Actionable recommendations for improvement
- A concise concluding summary

Feedback Summary:
{summary}

Student Satisfaction Report:
""",
    input_variables=["summary"]
)


report_chain = LLMChain(llm=llm, prompt=report_prompt)

# ---------------- SIDEBAR ----------------
role = st.sidebar.selectbox("Select Role", ["Student", "Admin"])

# ---------------- CHAT STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- CHAT INPUT (ONLY ONCE) ----------------
query = st.chat_input("Ask your question", key="main_chat_input")

if query:
    # store user message
    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    if role == "Admin" and "summarize" in query.lower():
        df = pd.read_csv(FEEDBACK_FILE)
        feedback_text = "\n".join(df["Feedback"].astype(str).tolist())
        response = feedback_chain.invoke({"feedback": feedback_text})["text"]

    elif role == "Admin" and "report" in query.lower():
        df = pd.read_csv(FEEDBACK_FILE)
        feedback_text = "\n".join(df["Feedback"].astype(str).tolist())
        summary = feedback_chain.invoke({"feedback": feedback_text})["text"]
        response = report_chain.invoke({"summary": summary})["text"]

    else:
        response = rag_chain.invoke({"query": query})["result"]

    # store assistant message
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
