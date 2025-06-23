import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Tu lógica RAG existente
from rag_system import RAGSystem  # Ajusta según tu estructura

app = FastAPI(
    title="Manual Mantenimiento API",
    description="API RAG para consultas del Manual de Mantenimiento de Salones del Reino",
    version="1.0.0"
)

# CORS para permitir requests desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG
rag_system = RAGSystem()

class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "message": "Manual Mantenimiento API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest):
    try:
        result = rag_system.query(request.query, request.user_id)
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
