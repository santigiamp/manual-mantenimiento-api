import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
import time

# Importar nuestro sistema RAG actualizado con imágenes reales
from rag_system import RemoteEmbeddingRAG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manual Mantenimiento API - Imágenes Reales",
    description="API RAG para consultas del Manual de Mantenimiento con imágenes REALES extraídas del PDF",
    version="3.0.0-real-images-from-pdf"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG con imágenes REALES
logger.info("🚀 Inicializando API con imágenes REALES del PDF...")
rag_system = RemoteEmbeddingRAG()

# Modelos de datos para imágenes reales
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"

class RealImageInfo(BaseModel):
    url: str
    description: str
    page: int
    filename: str = ""
    width: int = 0
    height: int = 0
    extracted_text: str = ""
    context: str = ""

class QueryResponse(BaseModel):
    answer: str
    images: List[RealImageInfo] = []  # Imágenes REALES del manual PDF
    sources: List[str] = []
    metadata: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """Eventos de inicio"""
    logger.info("✅ Manual Mantenimiento API - VERSIÓN IMÁGENES REALES")
    logger.info("🧠 Usando embeddings remotos (HuggingFace)")
    logger.info("🔍 Búsqueda vectorial en Qdrant Cloud")
    logger.info("🖼️ Imágenes REALES extraídas del PDF manual")
    logger.info("🔍 OCR aplicado para análisis contextual")
    logger.info("📸 Almacenamiento en ImgBB")
    
    # Verificar estado del RAG
    health = rag_system.health_check()
    if health["overall_status"] == "healthy":
        logger.info("✅ Sistema RAG completamente operacional con imágenes reales")
        logger.info("🎯 Listo para consultas con material visual del manual")
    else:
        logger.warning("⚠️ Sistema RAG no operacional")
        logger.warning("💡 Ejecuta el script de Kaggle para procesar el manual PDF")

@app.get("/")
async def root():
    return {
        "message": "Manual Mantenimiento API - Imágenes Reales",
        "status": "running",
        "version": "3.0.0-real-images-from-pdf",
        "docs": "/docs",
        "image_mode": "REAL_IMAGES_FROM_PDF",
        "features": {
            "vector_search": True,
            "remote_embeddings": True,
            "groq_llm": True,
            "real_image_extraction": True,
            "ocr_analysis": True,
            "imgbb_storage": True,
            "mock_images": False,
            "render_free_compatible": True
        },
        "endpoints": {
            "query": "/query",
            "health": "/health", 
            "rag_status": "/rag-status",
            "test_queries": "/test-queries",
            "secrets_check": "/secrets-check",
            "system_info": "/system-info",
            "images_by_page": "/images/{page_number}",
            "search_images": "/search-images"
        },
        "v3_features": [
            "🖼️ Imágenes REALES extraídas del PDF del manual",
            "🔍 OCR aplicado para análisis contextual de imágenes",
            "📸 URLs reales de ImgBB (no placeholders)",
            "🎯 Respuestas con material visual técnico auténtico",
            "📋 Metadatos completos: descripción, texto extraído, contexto",
            "✅ Eliminación completa de imágenes mock"
        ],
        "requirements": {
            "kaggle_processing": "Ejecutar script de Kaggle para extraer imágenes del PDF",
            "imgbb_account": "API key de ImgBB para almacenamiento de imágenes",
            "manual_pdf": "Manual de Mantenimiento (44 páginas) como fuente"
        }
    }

@app.get("/health")
async def health_check():
    """Health check básico de la API"""
    return {
        "status": "healthy",
        "service": "manual-mantenimiento-api-real-images",
        "api_status": "operational",
        "embedding_method": "remote_huggingface",
        "image_support": "REAL_IMAGES_FROM_PDF",
        "memory_optimized": True,
        "mock_free": True
    }

@app.get("/rag-status")
async def rag_status():
    """Estado detallado del sistema RAG con imágenes reales"""
    health = rag_system.health_check()
    
    return {
        "rag_system": health,
        "ready_for_queries": health["operational"],
        "image_mode": "REAL_IMAGES_FROM_PDF",
        "embedding_provider": "HuggingFace Inference API",
        "vector_database": "Qdrant Cloud",
        "llm_provider": "Groq",
        "image_storage": "ImgBB",
        "ocr_engine": "EasyOCR",
        "manual_processed": health["operational"],
        "mode": "real_images_full_rag" if health["operational"] else "not_processed",
        "processing_pipeline": [
            "📄 PDF → Extracción de texto por páginas",
            "🖼️ PDF → Extracción de imágenes (PyMuPDF)",
            "🔍 Imágenes → Análisis OCR (EasyOCR)",
            "📸 Imágenes → Upload a ImgBB",
            "🧠 Texto → Embeddings (HuggingFace)",
            "💾 Todo → Qdrant Cloud con metadatos completos"
        ],
        "advantages": [
            "✅ Imágenes auténticas del manual oficial",
            "✅ Descripciones contextuales precisas",
            "✅ Referencias visuales técnicamente correctas",
            "✅ Compatible con Render free plan", 
            "✅ Sin dependencias locales pesadas",
            "✅ URLs persistentes de ImgBB"
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest):
    """
    Endpoint principal para consultas del manual con imágenes REALES
    """
    start_time = time.time()
    
    try:
        logger.info(f"📝 Nueva consulta de {request.user_id}: {request.query[:100]}...")
        
        # Procesar consulta con el sistema RAG
        result = rag_system.query(
            query=request.query,
            user_id=request.user_id,
            include_images=True
        )
        
        # Procesar imágenes REALES
        real_images = []
        for img in result.get("images", []):
            # Validar que sea imagen real (no mock)
            if img.get('url') and not 'placeholder' in img.get('url', ''):
                real_images.append(RealImageInfo(
                    url=img['url'],
                    description=img['description'],
                    page=img['page'],
                    filename=img.get('filename', ''),
                    width=img.get('width', 0),
                    height=img.get('height', 0),
                    extracted_text=img.get('extracted_text', ''),
                    context=img.get('context', '')
                ))
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        # Construir respuesta
        response = QueryResponse(
            answer=result.get("answer", "Lo siento, no pude procesar tu consulta."),
            images=real_images,
            sources=result.get("sources", []),
            metadata={
                "query_processed": True,
                "embedding_method": "remote_huggingface",
                "search_method": "qdrant_vector_search",
                "image_mode": "REAL_IMAGES_FROM_PDF",
                "user_id": request.user_id,
                "response_time": response_time,
                "chunks_found": len(result.get("sources", [])),
                "real_images_found": len(real_images),
                "confidence_score": result.get("confidence_score", 0.0),
                "manual_pages_referenced": list(set([img.page for img in real_images])),
                "features_used": ["text_search", "real_image_support", "ocr_analysis"]
            }
        )
        
        logger.info(f"✅ Respuesta generada: {len(response.answer)} chars, {len(response.images)} imágenes REALES")
        if real_images:
            logger.info(f"🖼️ Imágenes incluidas: {[img.filename for img in real_images]}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta: {str(e)}"
        )

@app.get("/test-queries")
async def get_test_queries():
    """Consultas de prueba optimizadas para imágenes reales"""
    return {
        "test_queries": [
            {
                "category": "🔧 Equipos con Material Visual",
                "queries": [
                    "¿Cómo mantener las aspiradoras? Muéstrame el diagrama",
                    "¿Qué EPP usar para escaleras? Necesito ver las imágenes",
                    "¿Cómo revisar herramientas eléctricas? Con ilustraciones",
                    "Mantenimiento de máquinas - incluye material visual"
                ]
            },
            {
                "category": "🏢 Sistema de Emergencia con Diagramas",
                "queries": [
                    "¿Cómo revisar extintores? Muestra las partes del extintor",
                    "¿Qué incluye el sistema de emergencia? Con imágenes técnicas",
                    "¿Cómo probar iluminación de emergencia? Incluye diagramas",
                    "Señalización de escape - necesito ver los ejemplos visuales"
                ]
            },
            {
                "category": "🏗️ Inspecciones con Material Técnico",
                "queries": [
                    "¿Cómo inspeccionar estructuras? Con imágenes de referencia",
                    "¿Qué buscar en muros? Muestra ejemplos visuales de problemas",
                    "¿Cómo revisar techos? Incluye procedimientos ilustrados",
                    "Inspección post-desastre - con material visual de daños"
                ]
            },
            {
                "category": "⚡ Sistemas Eléctricos Ilustrados",
                "queries": [
                    "¿Cómo revisar tableros eléctricos? Con diagramas",
                    "¿Qué hacer con luminarias? Muestra el procedimiento visual",
                    "¿Cómo mantener puesta a tierra? Incluye esquemas",
                    "Distribución eléctrica - necesito ver los diagramas técnicos"
                ]
            },
            {
                "category": "🎵 Audio y Video con Esquemas",
                "queries": [
                    "¿Cómo limpiar equipos de audio? Con kit de limpieza ilustrado",
                    "¿Qué hacer si falla el micrófono? Incluye diagramas de conexión",
                    "¿Cómo mantener proyectores? Con procedimientos visuales",
                    "Sistema de sonido completo - muestra los componentes"
                ]
            },
            {
                "category": "❄️ Climatización con Ilustraciones",
                "queries": [
                    "¿Por qué gotea el aire acondicionado? Muestra el procedimiento de reparación",
                    "¿Cómo limpiar filtros del AC? Con pasos ilustrados",
                    "¿Qué hacer con ventiladores ruidosos? Incluye diagramas de componentes",
                    "Mantenimiento de climatización - con material visual completo"
                ]
            },
            {
                "category": "🚰 Sistema Hidráulico Ilustrado",
                "queries": [
                    "¿Cómo reparar canillas que gotean? Con procedimiento visual",
                    "¿Qué hacer con inodoros obstruidos? Muestra los pasos",
                    "¿Cómo mantener cañerías? Incluye diagramas del sistema",
                    "Problemas de presión de agua - con esquemas técnicos"
                ]
            },
            {
                "category": "🛠️ Reparaciones Rápidas Paso a Paso",
                "queries": [
                    "¿Cómo quitar óxido? Muestra el procedimiento completo con imágenes",
                    "¿Cómo reparar grietas en paredes? Con pasos visuales detallados",
                    "¿Qué hacer si se descascara la pintura? Incluye técnicas ilustradas",
                    "¿Cómo nivelar cielorraso? Con procedimiento visual paso a paso"
                ]
            },
            {
                "category": "🔒 Seguridad con EPP Visual",
                "queries": [
                    "¿Qué EPP usar? Muestra el diagrama completo del equipo",
                    "¿Cómo trabajar seguro en altura? Con ilustraciones de seguridad",
                    "¿Cuándo usar arnés? Incluye ejemplos visuales de situaciones",
                    "Seguridad eléctrica - con diagramas de bloqueo y etiquetado"
                ]
            },
            {
                "category": "🌱 Jardines y Mantenimiento Exterior",
                "queries": [
                    "¿Cómo mantener jardines? Con ejemplos visuales de herramientas",
                    "¿Qué EPP usar para jardinería? Muestra el equipo necesario",
                    "¿Cómo controlar plagas? Incluye identificación visual",
                    "Mantenimiento exterior completo - con material ilustrativo"
                ]
            }
        ],
        "image_specific_tips": [
            "🖼️ Pregunta específicamente por 'diagramas', 'imágenes' o 'material visual'",
            "📸 Las respuestas incluyen URLs reales de imágenes técnicas del manual",
            "🔍 Cada imagen tiene descripción detallada con contexto del manual",
            "📋 Se incluyen metadatos: página, dimensiones, texto extraído por OCR",
            "⚠️ Material visual auténtico del Manual oficial (no mocks)",
            "🎯 Imágenes complementan perfectamente las explicaciones técnicas"
        ],
        "usage_examples": [
            "💡 'Aire acondicionado que gotea con procedimiento visual'",
            "💡 'EPP completo - muestra todos los elementos'", 
            "💡 'Reparación de grietas paso a paso con imágenes'",
            "💡 'Sistema eléctrico - incluye diagramas técnicos'"
        ]
    }

@app.get("/secrets-check")
async def check_secrets():
    """Verificar variables de entorno para sistema con imágenes reales"""
    secrets_status = {}
    
    # Variables críticas
    critical_secrets = {
        "QDRANT_URL": "Qdrant Cloud URL",
        "QDRANT_API_KEY": "Qdrant API Key", 
        "GROQ_API_KEY": "Groq LLM API Key"
    }
    
    # Variables para imágenes reales
    image_secrets = {
        "IMGBB_API_KEY": "ImgBB para almacenamiento de imágenes reales",
        "HUGGINGFACE_API_KEY": "HuggingFace Embeddings (opcional)"
    }
    
    # Variables opcionales
    optional_secrets = {
        "SENTRY_DSN": "Sentry para logging (opcional)"
    }
    
    all_critical_good = True
    images_configured = False
    
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
            all_critical_good = False
    
    # Verificar secretos de imágenes
    for key, description in image_secrets.items():
        value = os.getenv(key)
        if value:
            secrets_status[key] = {
                "status": "✅ Configurado",
                "description": description,
                "required_for": "Procesamiento de imágenes reales"
            }
            if key == "IMGBB_API_KEY":
                images_configured = True
        else:
            secrets_status[key] = {
                "status": "⚠️ No configurado",
                "description": description,
                "required_for": "Procesamiento de imágenes reales"
            }
    
    # Verificar opcionales
    for key, description in optional_secrets.items():
        value = os.getenv(key)
        secrets_status[key] = {
            "status": "✅ Configurado" if value else "⚪ No configurado (opcional)",
            "description": description,
            "required": False
        }
    
    return {
        "overall_status": "✅ Sistema listo" if all_critical_good and images_configured else "❌ Configuración incompleta",
        "ready_for_production": all_critical_good and images_configured,
        "image_processing_ready": images_configured,
        "secrets": secrets_status,
        "processing_requirements": {
            "manual_pdf": "Manual de Mantenimiento (44 páginas)",
            "kaggle_script": "Script de procesamiento ejecutado",
            "imgbb_account": "Cuenta ImgBB para almacenamiento",
            "qdrant_collection": "Colección con chunks e imágenes"
        },
        "next_steps": [
            "1. Configura IMGBB_API_KEY si falta",
            "2. Sube manual_mantenimiento.pdf a Kaggle dataset",
            "3. Ejecuta el script de procesamiento en Kaggle",
            "4. Verifica que las imágenes se subieron a ImgBB",
            "5. Confirma que Qdrant tiene los chunks con metadatos",
            "6. Testea con /test-queries"
        ] if not (all_critical_good and images_configured) else [
            "🎉 ¡Sistema completamente configurado!",
            "📄 Procesa el manual en Kaggle si no lo hiciste",
            "🧪 Usa /test-queries para probar imágenes reales",
            "🖼️ Las consultas ahora incluyen material visual auténtico"
        ]
    }

@app.get("/system-info")
async def get_system_info():
    """Información detallada del sistema con imágenes reales"""
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
            "version": "3.0.0-real-images-from-pdf",
            "image_mode": "REAL_IMAGES_FROM_PDF",
            "embedding_method": "remote_huggingface",
            "vector_db": "qdrant_cloud",
            "llm_provider": "groq",
            "image_storage": "imgbb",
            "ocr_engine": "easyocr",
            "memory_optimized": True,
            "mock_free": True
        },
        "manual_info": {
            "title": "Manual de Mantenimiento - Salones del Reino",
            "pages": 44,
            "estimated_images": "20-30 diagramas e ilustraciones técnicas",
            "sections": [
                "01 - Introducción (EPP, Seguridad)",
                "02 - Equipos (Aspiradoras, Escaleras, Herramientas)", 
                "03 - Edificios (Emergencia, Inspecciones, Techos)",
                "04 - Sistemas Eléctricos (Distribución, Luminarias)",
                "05 - Sistemas Electrónicos (Audio, Video, Seguridad)", 
                "06 - Sistemas Mecánicos (Climatización, Agua)",
                "Anexo - Reparaciones Rápidas (Procedimientos Ilustrados)"
            ],
            "specialization": "Mantenimiento de infraestructura religiosa con soporte visual"
        },
        "image_processing_pipeline": {
            "step_1": "📄 Extracción de texto del PDF por páginas",
            "step_2": "🖼️ Extracción de imágenes usando PyMuPDF",
            "step_3": "🔍 Análisis OCR con EasyOCR (español/inglés)",
            "step_4": "📸 Upload de imágenes a ImgBB con nombres descriptivos",
            "step_5": "🧠 Generación de embeddings con HuggingFace",
            "step_6": "💾 Almacenamiento en Qdrant con metadatos completos",
            "step_7": "🎯 Asociación de imágenes con contexto textual"
        },
        "deployment": {
            "platform": "Railway/Render compatible",
            "free_tier_optimized": True,
            "external_dependencies": [
                "Qdrant Cloud (base vectorial)",
                "HuggingFace API (embeddings)",
                "Groq API (LLM)",
                "ImgBB (almacenamiento imágenes)",
                "Kaggle (procesamiento inicial)"
            ]
        },
        "advantages_over_mocks": [
            "✅ Imágenes auténticas del manual oficial",
            "✅ Descripciones contextuales precisas",
            "✅ OCR aplicado para mejor comprensión",
            "✅ URLs persistentes y confiables",
            "✅ Metadatos completos (página, dimensiones, contexto)",
            "✅ Referencias visuales técnicamente correctas"
        ]
    }

# Endpoints específicos para imágenes reales
@app.get("/images/{page_number}")
async def get_page_images(page_number: int):
    """Obtener imágenes REALES de una página específica del manual"""
    try:
        images = rag_system.get_images_by_page(page_number)
        
        # Filtrar solo imágenes reales
        real_images = [img for img in images if img.get('url') and not 'placeholder' in img.get('url', '')]
        
        return {
            "page": page_number,
            "images_count": len(real_images),
            "images": real_images,
            "manual_section": rag_system.get_section_by_page(page_number),
            "image_mode": "REAL_IMAGES_FROM_PDF"
        }
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo imágenes de página {page_number}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron imágenes reales para la página {page_number}"
        )

@app.get("/search-images")
async def search_images(query: str):
    """Buscar imágenes REALES relacionadas con una consulta"""
    try:
        images = rag_system.search_related_images(query)
        
        # Filtrar solo imágenes reales
        real_images = [img for img in images if img.get('url') and not 'placeholder' in img.get('url', '')]
        
        return {
            "query": query,
            "images_found": len(real_images),
            "images": real_images,
            "image_mode": "REAL_IMAGES_FROM_PDF",
            "note": "Solo se muestran imágenes auténticas extraídas del manual PDF"
        }
        
    except Exception as e:
        logger.error(f"❌ Error buscando imágenes para '{query}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error buscando imágenes reales: {str(e)}"
        )

@app.get("/processing-status")
async def check_processing_status():
    """Verificar si el manual fue procesado correctamente"""
    health = rag_system.health_check()
    
    return {
        "manual_processed": health["operational"],
        "image_extraction": "REAL_IMAGES_FROM_PDF" if health["operational"] else "NOT_PROCESSED",
        "qdrant_ready": health["qdrant_configured"],
        "processing_required": not health["operational"],
        "status": "✅ Manual procesado con imágenes reales" if health["operational"] else "❌ Manual no procesado",
        "instructions": [
            "1. Sube manual_mantenimiento.pdf a Kaggle dataset",
            "2. Configura variables de entorno en Kaggle",
            "3. Ejecuta el script kaggle_pdf_processor.py",
            "4. Verifica que las imágenes se subieron a ImgBB",
            "5. Confirma que Qdrant tiene los chunks completos"
        ] if not health["operational"] else [
            "🎉 Manual completamente procesado",
            "🖼️ Imágenes reales disponibles",
            "🧪 Sistema listo para consultas con material visual"
        ]
    }

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Error no manejado: {str(exc)}")
    return {
        "error": "Error interno del servidor",
        "detail": str(exc),
        "status_code": 500,
        "note": "Si el error persiste, verifica que el manual haya sido procesado en Kaggle"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
