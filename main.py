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

# REEMPLAZAR LA FUNCIÓN get_embedding_hf() EN main.py

def get_embedding_hf(text: str) -> List[float]:
    """Generar embedding usando HuggingFace con token o fallback"""
    try:
        # Opción 1: Usar API de HuggingFace con token
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if hf_token:
            url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json()
                result = embedding[0] if isinstance(embedding[0], list) else embedding
                logger.info(f"✅ HuggingFace embedding generado correctamente")
                return result
            else:
                logger.warning(f"⚠️ HuggingFace API falló: {response.status_code}, usando fallback")
        
        # Opción 2: Fallback - embedding simple basado en hash
        # Esto es temporal hasta que configures el token
        import hashlib
        import struct
        
        # Crear un embedding determinístico basado en el texto
        text_hash = hashlib.md5(text.encode()).digest()
        
        # Convertir a vector de 384 dimensiones (compatible con all-MiniLM-L6-v2)
        embedding = []
        for i in range(0, len(text_hash), 4):
            if i + 4 <= len(text_hash):
                value = struct.unpack('f', text_hash[i:i+4])[0]
                embedding.append(float(value))
        
        # Completar hasta 384 dimensiones
        while len(embedding) < 384:
            embedding.append(0.1)
        
        # Normalizar
        import math
        norm = math.sqrt(sum(x*x for x in embedding))
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        logger.info(f"⚠️ Usando embedding fallback para: '{text[:50]}...'")
        return embedding[:384]  # Asegurar exactamente 384 dimensiones
        
    except Exception as e:
        logger.error(f"❌ Error generando embedding: {str(e)}")
        # Embedding por defecto si todo falla
        return [0.1] * 384

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
            #score_threshold=0.05  # Umbral mínimo de similitud
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

# REEMPLAZAR generate_answer_groq() CON ESTA FUNCIÓN EN main.py

def generate_answer_hf(query: str, context_chunks: List[Dict]) -> str:
    """Generar respuesta usando HuggingFace gratis"""
    try:
        # Preparar contexto del manual
        context_text = "\n\n".join([
            f"**{chunk['section_name']} (Página {chunk['page']})**\n{chunk['content'][:500]}"
            for chunk in context_chunks[:3]
        ])
        
        # Prompt optimizado para el manual
        prompt = f"""Basándote únicamente en esta información del Manual de Mantenimiento, responde la consulta:

INFORMACIÓN DEL MANUAL:
{context_text}

CONSULTA: {query}

RESPUESTA (usa formato claro con emojis técnicos):"""

        # Usar modelo gratuito de HuggingFace
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        models_to_try = [
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill", 
            "microsoft/DialoGPT-medium",
            "google/flan-t5-large"
        ]
        
        for model in models_to_try:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {
                    "Content-Type": "application/json"
                }
                
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.3,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extraer respuesta según el modelo
                    if isinstance(result, list) and len(result) > 0:
                        answer = result[0].get('generated_text', '').strip()
                    elif isinstance(result, dict):
                        answer = result.get('generated_text', '').strip()
                    else:
                        continue
                    
                    if answer and len(answer) > 10:
                        logger.info(f"✅ Respuesta generada con {model}")
                        
                        # Formatear respuesta
                        formatted_answer = f"🔧 **Manual de Mantenimiento**\n\n{answer}\n\n"
                        
                        # Agregar referencias
                        pages = [str(chunk['page']) for chunk in context_chunks]
                        formatted_answer += f"📚 **Referencias:** Páginas {', '.join(pages)}"
                        
                        return formatted_answer
                        
                else:
                    logger.warning(f"⚠️ {model} falló: {response.status_code}")
                    continue
                    
            except Exception as e:
                logger.warning(f"⚠️ Error con {model}: {str(e)}")
                continue
        
        # Si todos los modelos fallan, usar fallback mejorado
        logger.warning("⚠️ Todos los modelos HF fallaron, usando fallback")
        return generate_enhanced_fallback(query, context_chunks)
        
    except Exception as e:
        logger.error(f"❌ Error generando respuesta HF: {str(e)}")
        return generate_enhanced_fallback(query, context_chunks)

def generate_enhanced_fallback(query: str, context_chunks: List[Dict]) -> str:
    """Fallback mejorado sin IA"""
    if not context_chunks:
        return f"❌ No encontré información sobre '{query}' en el manual."
    
    answer = f"🔧 **{query}**\n\n"
    
    for i, chunk in enumerate(context_chunks, 1):
        answer += f"📄 **{chunk['section_name']} (Página {chunk['page']})**\n"
        content = chunk['content'][:400].strip()
        answer += f"{content}...\n\n"
        
        if chunk.get('images'):
            answer += f"🖼️ Ver {len(chunk['images'])} imágenes técnicas en el manual\n\n"
    
    pages = [str(chunk['page']) for chunk in context_chunks]
    answer += f"📚 **Consultar páginas {', '.join(pages)} del manual para procedimientos completos**"
    
    return answer

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
