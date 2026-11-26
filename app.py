import tempfile
from pathlib import Path
import streamlit as st
from config import DATA_DIR, OPENAI_API_KEY
from rag.ingestion import ingest_path
from rag.vector_store import add_chunks
from rag.qa import answer_question
from rag.analytics import load_transactions, revenue_by_month, top_outstanding_customer, revenue_by_region

st.set_page_config(page_title="Finance AI Assistant", page_icon="📊", layout="wide")
st.title("📊 Enterprise Finance AI Assistant")
st.caption("RAG document Q&A + deterministic finance analytics | Synthetic demo data")

with st.sidebar:
    st.header("Knowledge base")
    if st.button("Index sample documents", use_container_width=True):
        chunks = []
        for path in sorted((DATA_DIR / "sample_docs").glob("*")):
            if path.is_file():
                chunks.extend(ingest_path(path))
        st.success(f"Indexed / updated {add_chunks(chunks)} chunks.")
    uploads = st.file_uploader("Upload PDF, TXT, MD or CSV", type=["pdf", "txt", "md", "csv"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads, use_container_width=True):
        chunks = []
        for uploaded in uploads:
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)
            try:
                new_chunks = ingest_path(tmp_path)
                for item in new_chunks:
                    item["metadata"]["source"] = uploaded.name
                chunks.extend(new_chunks)
            finally:
                tmp_path.unlink(missing_ok=True)
        st.success(f"Indexed {add_chunks(chunks)} chunks.")
    st.caption("LLM: " + ("Enabled" if OPENAI_API_KEY else "Retrieval-only; configure OPENAI_API_KEY in .env"))

qa_tab, analytics_tab, about_tab = st.tabs(["💬 Ask documents", "📈 Finance analytics", "ℹ️ About"])
with qa_tab:
    question = st.chat_input("Ask about a policy or report...")
    if question:
        with st.spinner("Retrieving and generating answer..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        with st.expander("Retrieved sources", expanded=True):
            for source in result["sources"]:
                st.markdown(f"**{source['source']}** · chunk {source['chunk']} · distance {source['distance']}")
                st.caption(source["excerpt"])
with analytics_tab:
    df = load_transactions(DATA_DIR / "finance_transactions.csv")
    months = sorted(df.invoice_date.dt.strftime("%Y-%m").unique())
    month = st.selectbox("Revenue month", months, index=len(months)-1)
    a, b = st.columns(2)
    a.metric("Revenue", f"£{revenue_by_month(df, month):,.2f}")
    top = top_outstanding_customer(df)
    b.metric("Highest outstanding customer", top["customer"], f"£{top['outstanding_balance']:,.2f}")
    st.bar_chart(revenue_by_region(df))
    st.dataframe(df, use_container_width=True, hide_index=True)
with about_tab:
    st.markdown("RAG handles unstructured document questions. Pandas handles exact financial calculations. All included data is fictional.")
