# NITK Academic Rules Q&A Chatbot (RAG)

Answers student questions on academic and hostel regulations from official NITK PDFs, with source citations.

**Stack:** Python, LangChain, Sentence-Transformers (`all-MiniLM-L6-v2`), ChromaDB, Streamlit, Groq / Gemini.

## How it works
1. PDFs in `docs/` are split into ~800-char overlapping chunks.
2. Chunks are embedded with Sentence-Transformers and stored in ChromaDB.
3. For each question, the top 4 similar chunks are retrieved.
4. The LLM answers only from that context and cites `[file, p.N]`; if the answer isn't there, it says so (reduces hallucination).

## Run locally
```bash
pip install -r requirements.txt
mkdir docs        # add official NITK PDFs (academic rules, hostel rules, etc.)

# choose one provider
export LLM_PROVIDER=groq   GROQ_API_KEY=your_key
# export LLM_PROVIDER=gemini GOOGLE_API_KEY=your_key

streamlit run app.py
```
Click **Re-index PDFs** in the sidebar the first time (and whenever PDFs change).

## Deploy (Streamlit Community Cloud)
1. Push this folder (including `docs/`) to GitHub.
2. Create the app on share.streamlit.io, pointing to `app.py`.
3. Add `LLM_PROVIDER` and the API key under **Secrets**.
4. Index once from the sidebar after first launch (or commit `chroma_db/`).

## Files
- `rag.py` - ingestion, retrieval, prompt, LLM selection
- `app.py` - Streamlit chat UI
