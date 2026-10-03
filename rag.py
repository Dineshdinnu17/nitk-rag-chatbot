"""Core RAG pipeline for the NITK Academic Rules Q&A chatbot."""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("docs")          # put official NITK PDFs here
DB_DIR = "chroma_db"             # persisted vector store
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for NITK Surathkal students answering questions about "
     "academic and hostel regulations. Answer ONLY from the context below. "
     "If the context does not contain the answer, say you could not find it in the "
     "official documents and suggest contacting the relevant office. "
     "Be concise, and cite sources inline as [filename, p.N].\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index() -> int:
    """Load PDFs, chunk them, embed with Sentence-Transformers, store in ChromaDB."""
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in ./{DOCS_DIR}/")

    pages = []
    for pdf in pdfs:
        for page in PyPDFLoader(str(pdf)).load():
            page.metadata["source"] = pdf.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
            pages.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    # Rebuild from scratch so re-indexing never duplicates chunks
    existing = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    existing.delete_collection()
    Chroma.from_documents(chunks, get_embeddings(), persist_directory=DB_DIR)
    return len(chunks)


def get_vectorstore() -> Chroma:
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())


def get_llm():
    """Pick the LLM via LLM_PROVIDER=groq|gemini (default groq)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )


def answer(question: str, vectorstore: Chroma, llm):
    """Retrieve relevant chunks, then generate a grounded answer with sources."""
    docs = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n".join(
        f"[{d.metadata['source']}, p.{d.metadata['page']}]\n{d.page_content}" for d in docs
    )
    reply = (PROMPT | llm).invoke({"context": context, "question": question})
    sources = [
        {"file": d.metadata["source"], "page": d.metadata["page"], "snippet": d.page_content[:300]}
        for d in docs
    ]
    return reply.content, sources
