import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import logging

# Importar nuestro sistema RAG
from rag_system import RAGSystem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manual Mantenimiento API",
    description="API RAG para consultas del Manual de Mantenimiento de Salones del Reino",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG
logger.info("Inicializando sistema RAG...")
rag_system = RAGSystem()

# Modelos de datos
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: Dict[str, Any] = {}

class DocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {
        "message": "Manual Mantenimiento API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/query", "/rag-status", "/add-documents"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "manual-mantenimiento-api",
        "api_status": "operational"
    }

@app.get("/rag-status")
async def rag_status():
    """Estado del sistema RAG"""
    return rag_system.health_check()

@app.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest):
    """Endpoint principal para consultas al manual"""
    try:
        logger.info(f"Nueva consulta: {request.query[:50]}... (usuario: {request.user_id})")
        
        # Procesar consulta con el sistema RAG
        result = rag_system.query(request.query, request.user_id)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando consulta: {str(e)}"
        )

@app.post("/add-documents")
async def add_documents(request: DocumentRequest):
    """Endpoint para agregar documentos al sistema RAG"""
    try:
        success = rag_system.add_documents(request.documents)
        if success:
            return {
                "message": f"Agregados {len(request.documents)} documentos exitosamente",
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Error agregando documentos")
            
    except Exception as e:
        logger.error(f"Error agregando documentos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-queries")
async def test_queries():
    """Endpoint para probar consultas comunes"""
    test_cases = [
        "¿Cómo reparar aire acondicionado que gotea?",
        "¿Cómo arreglar grietas en la pared?", 
        "¿Cómo pintar paredes dañadas?",
        "¿Qué mantenimiento necesita el sistema eléctrico?"
    ]
    
    results = {}
    for query in test_cases:
        try:
            result = rag_system.query(query, "test_user")
            results[query] = {
                "answer": result["answer"][:100] + "...",  # Solo primeros 100 chars
                "sources_count": len(result.get("sources", [])),
                "status": "success"
            }
        except Exception as e:
            results[query] = {
                "error": str(e),
                "status": "error"
            }
    
    return results

@app.get("/environment")
async def get_environment():
    """Ver configuración del entorno (sin exponer secrets)"""
    return {
        "qdrant_url_set": bool(os.getenv("QDRANT_URL")),
        "qdrant_api_key_set": bool(os.getenv("QDRANT_API_KEY")),
        "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
        "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "manual_mantenimiento"),
        "python_version": os.sys.version,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
