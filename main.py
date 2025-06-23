import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import logging

# Importar nuestro sistema RAG con embeddings remotos
from rag_system import RemoteEmbeddingRAG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manual Mantenimiento API",
    description="API RAG para consultas del Manual de Mantenimiento de Salones del Reino",
    version="2.0.0-remote-embeddings"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG con embeddings remotos
logger.info("🚀 Inicializando API con Remote Embeddings...")
rag_system = RemoteEmbeddingRAG()

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
    logger.info("✅ Manual Mantenimiento API iniciada exitosamente")
    logger.info("🧠 Usando embeddings remotos (HuggingFace)")
    logger.info("🔍 Búsqueda vectorial completa en Qdrant")
    
    # Verificar estado del RAG
    health = rag_system.health_check()
    if health["overall_status"] == "healthy":
        logger.info("✅ Sistema RAG completamente operacional")
    else:
        logger.warning("⚠️ Sistema RAG en modo limitado - usando respuestas mock")

@app.get("/")
async def root():
    return {
        "message": "Manual Mantenimiento API",
        "status": "running",
        "version": "2.0.0-remote-embeddings",
        "docs": "/docs",
        "features": {
            "vector_search": True,
            "remote_embeddings": True,
            "groq_llm": True,
            "render_free_compatible": True
        },
        "endpoints": {
            "query": "/query",
            "health": "/health", 
            "rag_status": "/rag-status",
            "test_queries": "/test-queries",
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
        "embedding_method": "remote_huggingface",
        "memory_optimized": True
    }

@app.get("/rag-status")
async def rag_status():
    """Estado detallado del sistema RAG"""
    health = rag_system.health_check()
    
    return {
        "rag_system": health,
        "ready_for_queries": health["operational"],
        "embedding_provider": "HuggingFace Inference API",
        "vector_database": "Qdrant Cloud",
        "llm_provider": "Groq",
        "mode": "remote_embeddings_full_rag" if health["operational"] else "mock_fallback",
        "advantages": [
            "✅ Búsqueda vectorial completa",
            "✅ Compatible con Render free plan", 
            "✅ Sin dependencias pesadas locales",
            "✅ Embeddings de calidad profesional"
        ]
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
            "1. Configure missing secrets in Render Environment Variables",
            "2. Redeploy the service",
            "3. Upload manual using Kaggle",
            "4. Test vector search functionality"
        ] if not all_configured else [
            "✅ All secrets configured",
            "🔄 Upload manual to Qdrant using Kaggle",
            "🧪 Test queries to verify functionality"
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest):
    """Endpoint principal para consultas al manual con búsqueda vectorial"""
    try:
        logger.info(f"🔍 Nueva consulta: {request.query[:50]}... (usuario: {request.user_id})")
        
        # Procesar consulta con el sistema RAG de embeddings remotos
        result = await rag_system.query(request.query, request.user_id)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"❌ Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando consulta: {str(e)}"
        )

@app.get("/test-queries")
async def test_queries():
    """Endpoint para probar consultas comunes con búsqueda vectorial"""
    test_cases = [
        "¿Cómo reparar aire acondicionado que gotea?",
        "¿Cómo arreglar grietas en la pared?", 
        "¿Cómo pintar paredes dañadas?",
        "¿Cómo desobstruir registros de agua?",
        "¿Qué mantenimiento necesitan las luminarias?",
        "¿Qué hacer con óxido en elementos metálicos?",
        "¿Cómo inspeccionar el sistema eléctrico?"
    ]
    
    results = {}
    for query in test_cases:
        try:
            result = await rag_system.query(query, "test_user")
            results[query] = {
                "answer_preview": result["answer"][:150] + "...",
                "sources_count": len(result.get("sources", [])),
                "status": "success",
                "search_method": result.get("metadata", {}).get("search_method", "unknown"),
                "system_status": result.get("metadata", {}).get("system_status", "unknown")
            }
        except Exception as e:
            results[query] = {
                "error": str(e),
                "status": "error"
            }
    
    successful_tests = len([r for r in results.values() if r.get("status") == "success"])
    
    return {
        "test_results": results,
        "summary": {
            "total_tests": len(test_cases),
            "successful_tests": successful_tests,
            "success_rate": f"{(successful_tests/len(test_cases)*100):.1f}%"
        },
        "embedding_method": "remote_huggingface",
        "vector_search": "qdrant_cloud"
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
        "api_version": "2.0.0-remote-embeddings",
        "features": {
            "rag_enabled": True,
            "embedding_method": "remote_huggingface",
            "vector_search": "qdrant_cloud",
            "llm_provider": "groq",
            "memory_optimized": True
        }
    }

@app.get("/system-info")
async def system_info():
    """Información técnica del sistema"""
    return {
        "architecture": {
            "embeddings": "HuggingFace Inference API (remote)",
            "vector_database": "Qdrant Cloud",
            "llm": "Groq (Llama 3.1 70B)",
            "api_framework": "FastAPI",
            "deployment": "Render (free plan compatible)"
        },
        "workflow": [
            "1. 👤 Usuario envía consulta",
            "2. 🔢 HuggingFace genera embedding remoto",
            "3. 🔍 Qdrant busca vectores similares",
            "4. 📋 Obtiene chunks relevantes del manual",
            "5. 🤖 Groq genera respuesta contextual",
            "6. ✅ Usuario recibe respuesta del manual"
        ],
        "benefits": [
            "✅ Búsqueda vectorial semántica completa",
            "✅ Compatible con planes gratuitos",
            "✅ Sin limitaciones de memoria local",
            "✅ Respuestas basadas en manual oficial",
            "✅ Escalable y mantenible"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
