import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import logging

# Importar nuestro sistema RAG con embeddings remotos y soporte de imágenes
from rag_system import RemoteEmbeddingRAG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manual Mantenimiento API",
    description="API RAG para consultas del Manual de Mantenimiento de Salones del Reino con soporte para imágenes",
    version="2.1.0-remote-embeddings-with-images"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG con embeddings remotos e imágenes
logger.info("🚀 Inicializando API con Remote Embeddings e imágenes...")
rag_system = RemoteEmbeddingRAG()

# Modelos de datos actualizados para incluir imágenes
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"

class ImageInfo(BaseModel):
    url: str
    description: str
    page: int
    filename: str = ""
    width: int = 0
    height: int = 0

class QueryResponse(BaseModel):
    answer: str
    images: List[ImageInfo] = []  # NUEVO: Lista de imágenes relevantes
    sources: List[str] = []
    metadata: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """Eventos de inicio"""
    logger.info("✅ Manual Mantenimiento API iniciada exitosamente")
    logger.info("🧠 Usando embeddings remotos (HuggingFace)")
    logger.info("🔍 Búsqueda vectorial completa en Qdrant")
    logger.info("🖼️ Soporte para imágenes activado")
    logger.info("🖼️ Soporte para imágenes activado")
    
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
        "version": "2.1.0-remote-embeddings-with-images",
        "docs": "/docs",
        "features": {
            "vector_search": True,
            "remote_embeddings": True,
            "groq_llm": True,
            "image_support": True,  # NUEVO
            "render_free_compatible": True
        },
        "endpoints": {
            "query": "/query",
            "health": "/health", 
            "rag_status": "/rag-status",
            "test_queries": "/test-queries",
            "secrets_check": "/secrets-check",
            "system_info": "/system-info"
        },
        "new_in_v2_1": [
            "🖼️ Extracción de imágenes del manual PDF",
            "🔗 URLs de imágenes en respuestas",
            "📸 Referencias visuales automáticas",
            "🎯 Respuestas más completas con material gráfico"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check básico de la API"""
    return {
        "status": "healthy",
        "service": "manual-mantenimiento-api",
        "api_status": "operational",
        "embedding_method": "remote_huggingface",
        "image_support": True,
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
        "image_storage": "ImgBB (via Kaggle upload)",
        "mode": "remote_embeddings_full_rag_with_images" if health["operational"] else "mock_fallback",
        "advantages": [
            "✅ Búsqueda vectorial completa",
            "✅ Compatible con Render free plan", 
            "✅ Sin dependencias locales pesadas",
            "✅ Embeddings remotos eficientes",
            "✅ Soporte completo de imágenes"
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest):
    """
    Endpoint principal para consultas del manual con soporte de imágenes
    """
    try:
        logger.info(f"📝 Nueva consulta de {request.user_id}: {request.query[:100]}...")
        
        # Procesar consulta con el sistema RAG
        result = rag_system.query(
            query=request.query,
            user_id=request.user_id,
            include_images=True  # NUEVO: Incluir imágenes en la respuesta
        )
        
        # Construir respuesta con imágenes
        response = QueryResponse(
            answer=result.get("answer", "Lo siento, no pude procesar tu consulta."),
            images=result.get("images", []),  # NUEVO: Lista de imágenes relevantes
            sources=result.get("sources", []),
            metadata={
                "query_processed": True,
                "embedding_method": "remote_huggingface",
                "search_method": "qdrant_vector_search",
                "image_support": True,
                "user_id": request.user_id,
                "response_time": result.get("response_time", 0),
                "chunks_found": len(result.get("sources", [])),
                "images_found": len(result.get("images", [])),  # NUEVO
                "confidence_score": result.get("confidence_score", 0.0)
            }
        )
        
        logger.info(f"✅ Respuesta generada: {len(response.answer)} chars, {len(response.images)} imágenes")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}"
        )

@app.get("/test-queries")
async def get_test_queries():
    """Consultas de prueba específicas del Manual de Mantenimiento"""
    return {
        "test_queries": [
            {
                "category": "🔧 Equipos",
                "queries": [
                    "¿Cómo mantener las aspiradoras?",
                    "¿Qué hacer si una escalera está dañada?",
                    "¿Cómo revisar herramientas eléctricas?",
                    "Mantenimiento de máquinas y herramientas"
                ]
            },
            {
                "category": "🏢 Edificios - Sistema de Emergencia",
                "queries": [
                    "¿Cómo revisar los extintores?",
                    "¿Cada cuánto probar la iluminación de emergencia?",
                    "¿Qué hacer si las salidas de escape están bloqueadas?",
                    "Mantenimiento de señalización de escape"
                ]
            },
            {
                "category": "🏗️ Inspecciones Estructurales",
                "queries": [
                    "¿Cómo inspeccionar columnas y vigas?",
                    "¿Qué buscar en muros de ladrillo?",
                    "¿Cómo revisar el techo después de una tormenta?",
                    "Inspección luego de un desastre natural"
                ]
            },
            {
                "category": "⚡ Sistemas Eléctricos",
                "queries": [
                    "¿Cómo revisar el tablero eléctrico?",
                    "¿Qué hacer si una lámpara no funciona?",
                    "¿Cómo mantener la puesta a tierra?",
                    "Mantenimiento de luminarias LED"
                ]
            },
            {
                "category": "🎵 Audio y Video",
                "queries": [
                    "¿Cómo limpiar los equipos de audio?",
                    "¿Qué hacer si el micrófono no funciona?",
                    "¿Cómo mantener el proyector?",
                    "Problemas con el sistema de sonido"
                ]
            },
            {
                "category": "❄️ Climatización",
                "queries": [
                    "¿Por qué gotea el aire acondicionado?",
                    "¿Cómo limpiar los filtros del AC?",
                    "¿Qué hacer si el ventilador hace ruido?",
                    "Mantenimiento de calefactores"
                ]
            },
            {
                "category": "🚰 Sistema Hidráulico",
                "queries": [
                    "¿Cómo reparar una canilla que gotea?",
                    "¿Qué hacer si el inodoro está obstruido?",
                    "¿Cómo mantener las cañerías?",
                    "Problemas con la presión del agua"
                ]
            },
            {
                "category": "🛠️ Reparaciones Rápidas",
                "queries": [
                    "¿Cómo quitar óxido de elementos metálicos?",
                    "¿Cómo reparar grietas en paredes?",
                    "¿Qué hacer si se descascara la pintura?",
                    "¿Cómo nivelar un cielorraso que se pandeó?"
                ]
            },
            {
                "category": "🌱 Jardines y Exterior",
                "queries": [
                    "¿Cómo mantener el césped del salón?",
                    "¿Qué EPP usar para jardinería?",
                    "¿Cómo controlar plagas en el edificio?",
                    "Mantenimiento de canteros y plantas"
                ]
            },
            {
                "category": "🔒 Seguridad",
                "queries": [
                    "¿Qué EPP usar para trabajar en altura?",
                    "¿Cómo trabajar seguro con electricidad?",
                    "¿Cuándo usar el formulario DC-85?",
                    "Normas de seguridad para mantenimiento"
                ]
            }
        ],
        "usage_tips": [
            "💡 Prueba preguntas específicas como '¿Cómo reparar goteo de aire acondicionado?'",
            "📋 Las respuestas incluyen pasos detallados del manual",
            "🖼️ Algunas respuestas incluyen imágenes de referencia",
            "📖 Se citan las páginas y secciones del manual",
            "⚠️ Se incluyen advertencias de seguridad cuando aplican"
        ]
    }

@app.get("/secrets-check")
async def check_secrets():
    """Verificar que las variables de entorno estén configuradas"""
    secrets_status = {}
    
    # Variables críticas para el funcionamiento
    critical_secrets = {
        "QDRANT_URL": "Qdrant Cloud URL",
        "QDRANT_API_KEY": "Qdrant API Key", 
        "GROQ_API_KEY": "Groq LLM API Key",
        "HUGGINGFACE_API_KEY": "HuggingFace Embeddings API Key"
    }
    
    # Variables opcionales para funcionalidades extra
    optional_secrets = {
        "IMGBB_API_KEY": "ImgBB para imágenes (opcional)",
        "SENTRY_DSN": "Sentry para logging (opcional)"
    }
    
    all_good = True
    
    # Verificar secretos críticos
    for key, description in critical_secrets.items():
        value = os.getenv(key)
        if value:
            secrets_status[key] = {
                "status": "✅ Configurado",
                "description": description,
                "length": len(value),
                "preview": f"{value[:8]}..." if len(value) > 8 else "***"
            }
        else:
            secrets_status[key] = {
                "status": "❌ Faltante",
                "description": description,
                "required": True
            }
            all_good = False
    
    # Verificar secretos opcionales
    for key, description in optional_secrets.items():
        value = os.getenv(key)
        if value:
            secrets_status[key] = {
                "status": "✅ Configurado",
                "description": description,
                "required": False
            }
        else:
            secrets_status[key] = {
                "status": "⚪ No configurado (opcional)",
                "description": description,
                "required": False
            }
    
    return {
        "overall_status": "✅ Todos los secretos críticos configurados" if all_good else "❌ Faltan secretos críticos",
        "ready_for_production": all_good,
        "secrets": secrets_status,
        "next_steps": [
            "1. Configura las variables faltantes en Railway/Render",
            "2. Procesa el manual PDF en Kaggle",
            "3. Carga los embeddings a Qdrant Cloud",
            "4. Testea las consultas con /test-queries"
        ] if not all_good else [
            "🎉 ¡Todo configurado correctamente!",
            "🔄 Procesa el manual en Kaggle si aún no lo hiciste",
            "🧪 Usa /test-queries para probar el sistema"
        ]
    }

@app.get("/system-info")
async def get_system_info():
    """Información detallada del sistema"""
    import platform
    import psutil
    
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor() or "Unknown"
        },
        "memory": {
            "total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "usage_percent": psutil.virtual_memory().percent
        },
        "api_info": {
            "version": "2.1.0-remote-embeddings-with-images",
            "embedding_method": "remote_huggingface",
            "vector_db": "qdrant_cloud",
            "llm_provider": "groq",
            "image_support": True,
            "memory_optimized": True
        },
        "manual_info": {
            "title": "Manual de Mantenimiento - Salones del Reino",
            "pages": 44,
            "sections": [
                "01 - Introducción",
                "02 - Equipos", 
                "03 - Edificios",
                "04 - Sistemas Eléctricos",
                "05 - Sistemas Electrónicos", 
                "06 - Sistemas Mecánicos",
                "Anexo - Reparaciones Rápidas"
            ],
            "specialization": "Mantenimiento de infraestructura religiosa"
        },
        "deployment": {
            "platform": "Railway/Render compatible",
            "free_tier_optimized": True,
            "external_dependencies": [
                "Qdrant Cloud (vector DB)",
                "HuggingFace API (embeddings)",
                "Groq API (LLM)",
                "ImgBB (image storage)"
            ]
        }
    }

# Endpoint adicional para manejo de imágenes
@app.get("/images/{page_number}")
async def get_page_images(page_number: int):
    """Obtener imágenes de una página específica del manual"""
    try:
        images = rag_system.get_images_by_page(page_number)
        
        return {
            "page": page_number,
            "images_count": len(images),
            "images": images,
            "manual_section": rag_system.get_section_by_page(page_number)
        }
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo imágenes de página {page_number}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron imágenes para la página {page_number}"
        )

@app.get("/search-images")
async def search_images(query: str):
    """Buscar imágenes relacionadas con una consulta"""
    try:
        images = rag_system.search_related_images(query)
        
        return {
            "query": query,
            "images_found": len(images),
            "images": images
        }
        
    except Exception as e:
        logger.error(f"❌ Error buscando imágenes para '{query}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error buscando imágenes: {str(e)}"
        )

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Error no manejado: {str(exc)}")
    return {
        "error": "Error interno del servidor",
        "detail": str(exc),
        "status_code": 500
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Desactivado para producción
        log_level="info"
    )
