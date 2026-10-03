import os

import streamlit as st

# Let Streamlit Cloud secrets act as environment variables
try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

from rag import DB_DIR, answer, build_index, get_llm, get_vectorstore  # noqa: E402

st.set_page_config(page_title="NITK Academic Rules Q&A", page_icon="🎓")
st.title("🎓 NITK Academic Rules Q&A")
st.caption("Ask about academic and hostel regulations. Answers are grounded in official PDFs.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    return get_vectorstore(), get_llm()


with st.sidebar:
    st.header("Knowledge base")
    if st.button("Re-index PDFs from ./docs"):
        with st.spinner("Chunking and embedding..."):
            n = build_index()
        st.cache_resource.clear()
        st.success(f"Indexed {n} chunks")
    st.markdown(f"LLM provider: `{os.getenv('LLM_PROVIDER', 'groq')}`")

if not os.path.isdir(DB_DIR):
    st.info("No index found. Add PDFs to `docs/` and click **Re-index** in the sidebar.")
    st.stop()

vectorstore, llm = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['file']}**, page {s['page']}")
                    st.caption(s["snippet"] + "...")

if question := st.chat_input("e.g. What is the minimum attendance required?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching regulations..."):
            reply, sources = answer(question, vectorstore, llm)
        st.markdown(reply)
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**{s['file']}**, page {s['page']}")
                st.caption(s["snippet"] + "...")

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
