from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from rag.ingestion import ingest_path
from rag.vector_store import add_chunks
from rag.qa import answer_question
from config import DATA_DIR
from rag.analytics import load_transactions, revenue_by_month

app = FastAPI(title="Enterprise Finance RAG Assistant", version="1.0.0")

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".csv"}:
        raise HTTPException(status_code=400, detail="Supported: PDF, TXT, MD, CSV")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = Path(tmp.name)
    try:
        chunks = ingest_path(path)
        for chunk in chunks:
            chunk["metadata"]["source"] = file.filename or "uploaded_file"
        return {"indexed_chunks": add_chunks(chunks), "filename": file.filename}
    finally:
        path.unlink(missing_ok=True)

@app.post("/ask")
def ask(payload: Question):
    return answer_question(payload.question, payload.k)

@app.get("/analytics/revenue")
def analytics_revenue(month: str):
    try:
        df = load_transactions(DATA_DIR / "finance_transactions.csv")
        return {"month": month, "revenue": revenue_by_month(df, month), "currency": "GBP"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
