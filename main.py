import os
import sys
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

# Inicializar sistema RAG al startup
logger.info("🚀 Inicializando Manual Mantenimiento API...")
rag_system = RAGSystem()

# Modelos de datos
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    metadata: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """Eventos de inicio"""
    logger.info("🎯 API iniciada exitosamente")
    
    # Verificar estado del RAG
    health = rag_system.health_check()
    if health["overall_status"] == "healthy":
        logger.info("✅ Sistema RAG completamente operacional")
    elif health["overall_status"] == "partial":
        logger.warning("⚠️ Sistema RAG parcialmente operacional")
    else:
        logger.warning("🔄 Sistema RAG en modo limitado - usando respuestas mock")

@app.get("/")
async def root():
    return {
        "message": "Manual Mantenimiento API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "rag_system": "enabled",
        "endpoints": {
            "query": "/query",
            "health": "/health", 
            "rag_status": "/rag-status",
            "secrets_check": "/secrets-check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check básico de la API"""
    return {
        "status": "healthy",
        "service": "manual-mantenimiento-api",
        "api_status": "operational",
        "timestamp": "2025-01-20"
    }

@app.get("/rag-status")
async def rag_status():
    """Estado detallado del sistema RAG"""
    health = rag_system.health_check()
    
    return {
        "rag_system": health,
        "ready_for_queries": health["overall_status"] in ["healthy", "partial"],
        "mode": "full_rag" if health["overall_status"] == "healthy" else "mock_fallback"
    }

@app.get("/secrets-check")
async def secrets_check():
    """Verificar que los secrets están configurados"""
    secrets = {
        "QDRANT_URL": bool(os.getenv("QDRANT_URL")),
        "QDRANT_API_KEY": bool(os.getenv("QDRANT_API_KEY")), 
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "QDRANT_COLLECTION_NAME": bool(os.getenv("QDRANT_COLLECTION_NAME"))
    }
    
    all_configured = all(secrets.values())
    
    return {
        "secrets_configured": secrets,
        "all_ready": all_configured,
        "missing_secrets": [key for key, value in secrets.items() if not value],
        "status": "ready" if all_configured else "incomplete",
        "next_steps": [
            "Configure missing secrets in Render Environment Variables",
            "Redeploy the service",
            "Upload manual using Kaggle"
        ] if not all_configured else ["Upload manual to Qdrant using Kaggle"]
    }

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

@app.get("/test-queries")
async def test_queries():
    """Endpoint para probar consultas comunes"""
    test_cases = [
        "¿Cómo reparar aire acondicionado que gotea?",
        "¿Cómo arreglar grietas en la pared?", 
        "¿Cómo pintar paredes dañadas?",
        "¿Qué mantenimiento necesita el sistema eléctrico?",
        "¿Cómo limpiar luminarias?",
        "¿Qué hacer con registros obstruidos?"
    ]
    
    results = {}
    for query in test_cases:
        try:
            result = rag_system.query(query, "test_user")
            results[query] = {
                "answer_preview": result["answer"][:150] + "...",
                "sources_count": len(result.get("sources", [])),
                "status": "success",
                "mode": result.get("metadata", {}).get("system_status", "unknown")
            }
        except Exception as e:
            results[query] = {
                "error": str(e),
                "status": "error"
            }
    
    return {
        "test_results": results,
        "total_tests": len(test_cases),
        "successful_tests": len([r for r in results.values() if r.get("status") == "success"])
    }

@app.get("/environment")
async def get_environment():
    """Ver configuración del entorno (SIN exponer secrets)"""
    return {
        "secrets_status": {
            "qdrant_url_configured": bool(os.getenv("QDRANT_URL")),
            "qdrant_api_key_configured": bool(os.getenv("QDRANT_API_KEY")),
            "groq_api_key_configured": bool(os.getenv("GROQ_API_KEY")),
        },
        "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "manual_mantenimiento"),
        "python_version": sys.version.split()[0],
        "environment": os.getenv("ENVIRONMENT", "production"),
        "api_version": "1.0.0",
        "rag_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
