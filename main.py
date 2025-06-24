# ========================================
# FASTAPI RAG - MANUAL DE MANTENIMIENTO
# ========================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import requests
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import uvicorn
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURACIÓN
# ========================================

app = FastAPI(
    title="Manual de Mantenimiento RAG API",
    description="API para consultas especializadas del Manual de Mantenimiento",
    version="1.0.0"
)

# CORS para permitir requests desde N8N
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables de entorno
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "manual_mantenimiento_secciones")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Cliente Qdrant global
qdrant_client = None

def init_qdrant():
    """Inicializar cliente Qdrant"""
    global qdrant_client
    try:
        if QDRANT_URL and QDRANT_API_KEY:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            logger.info(f"✅ Conectado a Qdrant: {COLLECTION_NAME}")
            return True
        else:
            logger.error("❌ Variables Qdrant faltantes")
            return False
    except Exception as e:
        logger.error(f"❌ Error conectando Qdrant: {str(e)}")
        return False

# ========================================
# MODELOS PYDANTIC
# ========================================

class QueryRequest(BaseModel):
    query: str
    section_filter: Optional[str] = None
    include_images: bool = True
    max_results: int = 3
    user_id: Optional[str] = None

class ImageInfo(BaseModel):
    filename: str
    image_url: str
    description: str
    page: int

class SearchResult(BaseModel):
    content: str
    page: int
    section_name: str
    title: str
    score: float
    images: List[ImageInfo]
    specialization: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]
    processing_time: float
    total_sources: int
    has_images: bool
    section_focused: Optional[str] = None

# ========================================
# FUNCIONES CORE
# ========================================

def get_embedding_hf(text: str) -> List[float]:
    """Generar embedding usando HuggingFace"""
    try:
        url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()
            result = embedding[0] if isinstance(embedding[0], list) else embedding
            return result
        elif response.status_code == 503:
            # Modelo cargándose, esperar y reintentar
            import time
            time.sleep(3)
            response = requests.post(url, headers=headers, json={"inputs": text})
            if response.status_code == 200:
                embedding = response.json()
                return embedding[0] if isinstance(embedding[0], list) else embedding
        
        logger.error(f"❌ Error HuggingFace: {response.status_code}")
        return [0.0] * 384
        
    except Exception as e:
        logger.error(f"❌ Error generando embedding: {str(e)}")
        return [0.0] * 384

def search_similar_chunks(query: str, section_filter: str = None, limit: int = 5) -> List[Dict]:
    """Buscar chunks similares en Qdrant"""
    try:
        if not qdrant_client:
            raise Exception("Cliente Qdrant no inicializado")
        
        # Generar embedding de la consulta
        query_embedding = get_embedding_hf(query)
        
        # Configurar filtro por sección si se especifica
        search_filter = None
        if section_filter:
            search_filter = Filter(
                must=[FieldCondition(key="section_id", match=MatchValue(value=section_filter))]
            )
        
        # Realizar búsqueda vectorial
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            score_threshold=0.3  # Umbral mínimo de similitud
        )
        
        # Procesar resultados
        processed_results = []
        for hit in search_results:
            payload = hit.payload
            
            # Procesar imágenes
            images = []
            if payload.get('images'):
                for img in payload['images']:
                    images.append({
                        'filename': img.get('filename', 'imagen'),
                        'image_url': img.get('image_url', ''),
                        'description': img.get('description', ''),
                        'page': img.get('page', payload.get('page', 0))
                    })
            
            processed_results.append({
                'content': payload.get('content', ''),
                'page': payload.get('page', 0),
                'section_name': payload.get('section_name', 'Sección'),
                'title': payload.get('title', 'Sin título'),
                'score': float(hit.score),
                'images': images,
                'specialization': payload.get('specialization', 'Mantenimiento'),
                'section_id': payload.get('section_id', '')
            })
        
        return processed_results
        
    except Exception as e:
        logger.error(f"❌ Error en búsqueda: {str(e)}")
        return []

def generate_answer_groq(query: str, context_chunks: List[Dict]) -> str:
    """Generar respuesta usando Groq"""
    try:
        if not GROQ_API_KEY:
            return generate_fallback_answer(query, context_chunks)
        
        # Preparar contexto
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n--- FUENTE {i} ---\n"
            context_text += f"Sección: {chunk['section_name']}\n"
            context_text += f"Página: {chunk['page']}\n"
            context_text += f"Contenido: {chunk['content'][:800]}...\n"
            
            # Agregar info de imágenes si las hay
            if chunk.get('images'):
                context_text += f"Imágenes disponibles: {len(chunk['images'])} imágenes técnicas\n"
        
        # Prompt especializado
        system_prompt = """Eres un experto en mantenimiento de Salones del Reino. Respondes consultas basándote ÚNICAMENTE en el manual oficial.

INSTRUCCIONES:
1. Usa SOLO la información del manual proporcionada
2. Sé específico y técnico pero comprensible
3. Menciona números de página cuando sea relevante
4. Si hay imágenes disponibles, menciónalas como "Ver imágenes técnicas"
5. Estructura la respuesta de manera clara
6. Si no tienes suficiente información, dilo claramente

FORMATO DE RESPUESTA:
🔧 [Título de la respuesta]

[Respuesta técnica detallada]

📄 Fuentes: Páginas [números]
🖼️ [Mencionar si hay imágenes disponibles]"""

        user_prompt = f"""CONSULTA: {query}

CONTEXTO DEL MANUAL:
{context_text}

Responde la consulta basándote únicamente en la información del manual."""

        # Llamada a Groq
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content']
            return answer
        else:
            logger.error(f"❌ Error Groq: {response.status_code}")
            return generate_fallback_answer(query, context_chunks)
            
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {str(e)}")
        return generate_fallback_answer(query, context_chunks)

def generate_fallback_answer(query: str, context_chunks: List[Dict]) -> str:
    """Respuesta de emergencia sin LLM"""
    if not context_chunks:
        return f"❌ No encontré información específica sobre '{query}' en el manual de mantenimiento."
    
    answer = f"🔧 Información sobre: {query}\n\n"
    
    for i, chunk in enumerate(context_chunks, 1):
        answer += f"📄 **Fuente {i} - {chunk['section_name']} (Página {chunk['page']})**\n"
        answer += f"{chunk['content'][:400]}...\n\n"
        
        if chunk.get('images'):
            answer += f"🖼️ Imágenes disponibles: {len(chunk['images'])} diagramas técnicos\n\n"
    
    pages = [str(chunk['page']) for chunk in context_chunks]
    answer += f"📚 Consulta las páginas {', '.join(pages)} del manual para más detalles."
    
    return answer

# ========================================
# ENDPOINTS
# ========================================

@app.on_event("startup")
async def startup_event():
    """Inicializar conexiones al arrancar"""
    logger.info("🚀 Iniciando API de Manual de Mantenimiento...")
    init_qdrant()

@app.get("/")
async def root():
    """Endpoint de salud"""
    return {
        "service": "Manual de Mantenimiento RAG API",
        "status": "operational",
        "version": "1.0.0",
        "qdrant_connected": qdrant_client is not None,
        "collection": COLLECTION_NAME
    }

@app.get("/health")
async def health_check():
    """Verificación de salud completa"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "qdrant": qdrant_client is not None,
            "groq": GROQ_API_KEY is not None,
            "collection": COLLECTION_NAME
        }
    }
    
    # Verificar conexión a Qdrant
    if qdrant_client:
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            health_status["services"]["qdrant_details"] = {
                "vectors": collection_info.points_count,
                "status": "connected"
            }
        except:
            health_status["services"]["qdrant"] = False
            health_status["status"] = "degraded"
    
    return health_status

@app.post("/query", response_model=QueryResponse)
async def query_manual(request: QueryRequest):
    """Endpoint principal para consultas al manual"""
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Nueva consulta: {request.query}")
        
        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Servicio Qdrant no disponible")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Consulta vacía")
        
        # Buscar chunks relevantes
        search_results = search_similar_chunks(
            query=request.query,
            section_filter=request.section_filter,
            limit=request.max_results
        )
        
        if not search_results:
            return QueryResponse(
                query=request.query,
                answer="❌ No encontré información relevante sobre tu consulta en el manual.",
                sources=[],
                processing_time=0.0,
                total_sources=0,
                has_images=False,
                section_focused=request.section_filter
            )
        
        # Generar respuesta con LLM
        answer = generate_answer_groq(request.query, search_results)
        
        # Procesar fuentes para respuesta
        sources = []
        has_images = False
        
        for result in search_results:
            images = []
            if request.include_images and result.get('images'):
                for img in result['images']:
                    images.append(ImageInfo(**img))
                    has_images = True
            
            sources.append(SearchResult(
                content=result['content'],
                page=result['page'],
                section_name=result['section_name'],
                title=result['title'],
                score=result['score'],
                images=images,
                specialization=result['specialization']
            ))
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log para monitoreo
        logger.info(f"✅ Consulta procesada: {len(sources)} fuentes, {processing_time:.2f}s")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            total_sources=len(sources),
            has_images=has_images,
            section_focused=request.section_filter
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error procesando consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/sections")
async def get_sections():
    """Obtener lista de secciones disponibles"""
    sections = {
        "01_introduccion": "Introducción y Seguridad",
        "02_equipos": "Equipos",
        "03_edificios": "Edificios e Infraestructura",
        "04_electricos": "Sistemas Eléctricos",
        "05_electronicos": "Sistemas Electrónicos",
        "06_mecanicos": "Sistemas Mecánicos",
        "07_reparaciones": "Reparaciones Rápidas"
    }
    return {"sections": sections}

@app.get("/stats")
async def get_stats():
    """Estadísticas de la base de conocimiento"""
    try:
        if not qdrant_client:
            return {"error": "Qdrant no disponible"}
        
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        # Obtener muestra de datos
        sample = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10,
            with_payload=True
        )[0]
        
        # Contar chunks con imágenes
        chunks_with_images = 0
        total_images = 0
        
        for point in sample:
            if point.payload.get('has_images'):
                chunks_with_images += 1
                total_images += point.payload.get('image_count', 0)
        
        return {
            "total_vectors": collection_info.points_count,
            "vector_dimension": collection_info.config.params.vectors.size,
            "sample_chunks_with_images": chunks_with_images,
            "sample_total_images": total_images,
            "collection_name": COLLECTION_NAME
        }
        
    except Exception as e:
        return {"error": str(e)}

# ========================================
# CONFIGURACIÓN PARA DEPLOYMENT
# ========================================

if __name__ == "__main__":
    # Para desarrollo local
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
