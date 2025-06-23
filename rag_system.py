import os
import logging
import httpx
import json
from typing import List, Dict, Any, Optional
import uuid
import asyncio

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteEmbeddingRAG:
    def __init__(self):
        """RAG System usando embeddings remotos con soporte para imágenes"""
        try:
            # Configuración desde variables de entorno
            self.qdrant_url = os.getenv("QDRANT_URL")
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY") 
            self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "manual_mantenimiento")
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            
            # URLs para APIs externas
            self.qdrant_base_url = f"{self.qdrant_url.rstrip('/')}/collections/{self.collection_name}"
            
            # Log de configuración (sin exponer secrets)
            logger.info("🚀 Inicializando Remote Embedding RAG System con soporte para imágenes...")
            logger.info(f"✅ QDRANT_URL configurada: {bool(self.qdrant_url)}")
            logger.info(f"✅ QDRANT_API_KEY configurada: {bool(self.qdrant_api_key)}")
            logger.info(f"✅ GROQ_API_KEY configurada: {bool(self.groq_api_key)}")
            logger.info(f"📦 Collection: {self.collection_name}")
            logger.info("🖼️ Soporte para imágenes: Activado")
            
            # Validar configuración
            missing_vars = []
            if not self.qdrant_url:
                missing_vars.append("QDRANT_URL")
            if not self.qdrant_api_key:
                missing_vars.append("QDRANT_API_KEY")
            if not self.groq_api_key:
                missing_vars.append("GROQ_API_KEY")
                
            if missing_vars:
                logger.warning(f"⚠️ Variables faltantes: {', '.join(missing_vars)}")
                logger.warning("🔄 Sistema funcionará en modo limitado con respuestas mock")
                self.operational = False
            else:
                logger.info("✅ Todas las configuraciones listas")
                self.operational = True
                
        except Exception as e:
            logger.error(f"❌ Error inicializando RAG: {str(e)}")
            self.operational = False
    
    async def get_remote_embedding(self, text: str) -> List[float]:
        """Obtener embedding usando HuggingFace Inference API (GRATIS)"""
        try:
            logger.debug(f"🔢 Generando embedding para: {text[:50]}...")
            
            # Usar HuggingFace Inference API - mismo modelo que usábamos localmente
            url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            
            # Si no tienes token, usa sin autorización (limitado pero funciona)
            headers = {"Content-Type": "application/json"}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json={"inputs": text, "options": {"wait_for_model": True}}
                )
                
                if response.status_code == 200:
                    embedding = response.json()
                    
                    # HuggingFace devuelve lista de listas, tomamos la primera
                    if isinstance(embedding, list) and len(embedding) > 0:
                        if isinstance(embedding[0], list):
                            result = embedding[0]  # Tomar primera embedding
                        else:
                            result = embedding
                        
                        logger.debug(f"✅ Embedding generado: {len(result)} dimensiones")
                        return result
                    else:
                        raise ValueError("Formato de embedding inesperado")
                        
                elif response.status_code == 503:
                    logger.warning("⏳ Modelo cargándose, reintentando...")
                    await asyncio.sleep(2)
                    # Reintentar una vez
                    response = await client.post(url, headers=headers, json={"inputs": text})
                    if response.status_code == 200:
                        embedding = response.json()
                        return embedding[0] if isinstance(embedding[0], list) else embedding
                    else:
                        raise Exception(f"Error después de reintento: {response.status_code}")
                else:
                    raise Exception(f"Error HuggingFace: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {str(e)}")
            # Fallback: usar embedding mock (ceros)
            logger.warning("🔄 Usando embedding mock para fallback")
            return [0.0] * 384  # all-MiniLM-L6-v2 tiene 384 dimensiones
    
    async def search_qdrant_vectors(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        """Buscar vectores similares en Qdrant usando REST API"""
        try:
            if not self.operational:
                logger.warning("🔄 Sistema no operacional, usando respuestas mock")
                return self._get_mock_results("query")
            
            logger.debug(f"🔍 Buscando en Qdrant: {len(query_embedding)} dim vector")
            
            # URL para búsqueda vectorial en Qdrant
            search_url = f"{self.qdrant_base_url}/points/search"
            
            headers = {
                "api-key": self.qdrant_api_key,
                "Content-Type": "application/json"
            }
            
            search_payload = {
                "vector": query_embedding,
                "limit": limit,
                "with_payload": True,
                "with_vector": False  # No necesitamos los vectores de vuelta
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    search_url,
                    headers=headers,
                    json=search_payload
                )
                
                if response.status_code == 200:
                    search_result = response.json()
                    results = []
                    
                    for hit in search_result.get("result", []):
                        payload = hit.get("payload", {})
                        results.append({
                            'content': payload.get('content', ''),
                            'page': payload.get('page', 0),
                            'section': payload.get('section', ''),
                            'title': payload.get('title', ''),
                            'score': hit.get('score', 0.0),
                            'has_images': payload.get('has_images', False),
                            'image_count': payload.get('image_count', 0),
                            'images': payload.get('images', [])  # URLs e info de imágenes
                        })
                    
                    logger.info(f"✅ Encontrados {len(results)} chunks relevantes")
                    # Log de imágenes encontradas
                    total_images = sum(len(r.get('images', [])) for r in results)
                    if total_images > 0:
                        logger.info(f"🖼️ Total imágenes relevantes: {total_images}")
                    
                    return results
                    
                elif response.status_code == 404:
                    logger.warning("📋 Colección no existe en Qdrant, usando respuestas mock")
                    return self._get_mock_results("query")
                else:
                    logger.error(f"❌ Error Qdrant: {response.status_code} - {response.text}")
                    return self._get_mock_results("query")
                    
        except Exception as e:
            logger.error(f"❌ Error en búsqueda vectorial: {str(e)}")
            return self._get_mock_results("query")
    
    def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Respuestas mock del manual cuando Qdrant no está disponible"""
        query_lower = query.lower() if query else ""
        
        # Base de respuestas del manual real con imágenes mock
        if "aire acondicionado" in query_lower or "gotea" in query_lower or "ac" in query_lower:
            return [{
                'content': """AIRE ACONDICIONADO: Gotea la unidad interior

SOLUCIÓN:
1. Identificar la punta de la manguera de desagüe del equipo y verificar que no esté obstruida en la salida
2. Si el problema no está allí, introducir una cinta pasacable a través de la manguera hasta lograr desobstruirla
3. En algún punto de este proceso debería comenzar a salir agua por la manguera

CAUSA: Obstrucciones en el sistema de drenaje del condensado

IMAGENES EN ESTA PÁGINA:
- Imagen 1: Ilustración de procedimiento: Reparación de aire acondicionado (URL: https://via.placeholder.com/600x400/0066cc/ffffff?text=Aire+Acondicionado)""",
                'page': 43,
                'section': 'G',
                'title': 'Reparaciones Rápidas - Aire Acondicionado',
                'score': 0.95,
                'has_images': True,
                'image_count': 1,
                'images': [{
                    'image_url': 'https://via.placeholder.com/600x400/0066cc/ffffff?text=Aire+Acondicionado+Goteo',
                    'filename': 'manual_p43_aire_acondicionado',
                    'width': 600,
                    'height': 400,
                    'description': 'Ilustración de procedimiento: Reparación de aire acondicionado que gotea'
                }]
            }]
        
        elif "pintura" in query_lower or "pared" in query_lower or "marcas" in query_lower:
            return [{
                'content': """PAREDES: Pintura dañada (marcas, raspones, rayones, manchas)

PROCEDIMIENTO:
1. Limpiar la superficie con cepillo o trapo
2. Restaurar rayones y marcas con masilla o enduido
3. Lijar dejando la superficie alisada y uniforme
4. Eliminar todo el polvo y aplicar fijador/sellador
5. Dependiendo del tamaño de la superficie a cubrir, pintar con pincel o rodillo

IMAGENES EN ESTA PÁGINA:
- Imagen 1: Proceso de reparación de pintura paso a paso (URL: https://via.placeholder.com/600x400/cc6600/ffffff?text=Reparacion+Pintura)""",
                'page': 42,
                'section': 'E',
                'title': 'Reparaciones Rápidas - Pintura',
                'score': 0.92,
                'has_images': True,
                'image_count': 1,
                'images': [{
                    'image_url': 'https://via.placeholder.com/600x400/cc6600/ffffff?text=Reparacion+Pintura',
                    'filename': 'manual_p42_pintura_reparacion',
                    'width': 600,
                    'height': 400,
                    'description': 'Proceso de reparación de pintura paso a paso'
                }]
            }]
        
        elif "grieta" in query_lower or "rajadura" in query_lower or "fisura" in query_lower:
            return [{
                'content': """MURO DE LADRILLOS: Rajaduras y grietas

Para grietas pequeñas y fisuras que sólo afecten los revoques o la pintura:
1. Introducir el canto de una espátula en la grieta y abrirla en forma de "V"
2. Con un pincel seco dejar el interior de la grieta completamente libre de polvo
3. Rellenar la grieta con un material apropiado para el lugar (masilla plástica interior o exterior)
4. Dejar secar y lijar con un taco de lija hasta nivelar la pared

IMPORTANTE: Si se observa que las grietas son más profundas y afectan la estructura, comunicarse con el formador de mantenimiento.

IMAGENES EN ESTA PÁGINA:
- Imagen 1: Técnica de reparación de grietas en muros (URL: https://via.placeholder.com/600x400/cc0066/ffffff?text=Reparacion+Grietas)""",
                'page': 40,
                'section': 'B',
                'title': 'Reparaciones Rápidas - Grietas',
                'score': 0.89,
                'has_images': True,
                'image_count': 1,
                'images': [{
                    'image_url': 'https://via.placeholder.com/600x400/cc0066/ffffff?text=Reparacion+Grietas',
                    'filename': 'manual_p40_grietas_reparacion',
                    'width': 600,
                    'height': 400,
                    'description': 'Técnica de reparación de grietas en muros'
                }]
            }]
        
        elif "epp" in query_lower or "protección" in query_lower or "seguridad" in query_lower:
            return [{
                'content': """EQUIPO DE PROTECCIÓN PERSONAL (EPP)

El objetivo es lograr un entorno de trabajo sin accidentes. Usar siempre el EPP:

- Casco y gafas de protección
- Protector facial y auditivo
- Guantes de trabajo
- Chaleco reflectivo
- Arnés y faja lumbar
- Mameluco y calzado de seguridad
- Barbijo

IMAGENES EN ESTA PÁGINA:
- Imagen 1: Diagrama de Equipo de Protección Personal (EPP) (URL: https://via.placeholder.com/800x600/006600/ffffff?text=EPP+Completo)""",
                'page': 5,
                'section': 'introducción',
                'title': 'Introducción - Seguridad Personal',
                'score': 0.94,
                'has_images': True,
                'image_count': 1,
                'images': [{
                    'image_url': 'https://via.placeholder.com/800x600/006600/ffffff?text=EPP+Completo',
                    'filename': 'manual_p5_epp_diagrama',
                    'width': 800,
                    'height': 600,
                    'description': 'Diagrama técnico detallado: Equipo de Protección Personal (EPP)'
                }]
            }]
        
        else:
            return [{
                'content': """MANUAL DE MANTENIMIENTO - Salones del Reino

Este manual contiene información completa sobre:
• Seguridad personal y programa de mantenimiento
• Equipos (aspiradoras, escaleras, herramientas)
• Edificios (sistemas de emergencia, inspecciones, techos)
• Sistemas eléctricos (distribución, luminarias)
• Sistemas electrónicos (audio, video, seguridad)
• Sistemas mecánicos (climatización, dispensadores, agua)
• Reparaciones rápidas (óxido, grietas, pintura, selladores)

Para consultas específicas, pregunta sobre temas como: aire acondicionado, pintura, grietas, registros obstruidos, luminarias, sistemas eléctricos, EPP, etc.""",
                'page': 1,
                'section': 'Introducción',
                'title': 'Manual de Mantenimiento',
                'score': 0.7,
                'has_images': False,
                'image_count': 0,
                'images': []
            }]
    
    async def generate_answer_with_groq(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar respuesta usando Groq LLM incluyendo información de imágenes"""
        try:
            if not self.groq_api_key or not context_chunks:
                return self._format_mock_answer_with_images(context_chunks[0] if context_chunks else {})
            
            # Preparar contexto incluyendo imágenes
            context_parts = []
            relevant_images = []
            
            for chunk in context_chunks:
                context_parts.append(f"**{chunk['title']} (Página {chunk['page']})**\n{chunk['content']}")
                
                # Recopilar imágenes relevantes
                if chunk.get('images'):
                    for img in chunk['images']:
                        relevant_images.append({
                            'url': img['image_url'],
                            'description': img['description'],
                            'page': chunk['page'],
                            'filename': img.get('filename', ''),
                            'width': img.get('width', 0),
                            'height': img.get('height', 0)
                        })
            
            context = "\n\n".join(context_parts)
            
            # Indicar si hay material visual disponible
            has_images = len(relevant_images) > 0
            image_note = f"\n\n📸 MATERIAL VISUAL DISPONIBLE: {len(relevant_images)} imagen(es) técnica(s)" if has_images else ""
            
            # Prompt optimizado para incluir referencias a imágenes
            prompt = f"""Eres un asistente experto en mantenimiento de Salones del Reino. 
Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto del manual proporcionado.

**CONTEXTO DEL MANUAL:**
{context}{image_note}

**PREGUNTA DEL USUARIO:**
{query}

**INSTRUCCIONES:**
- Responde en español con información práctica y específica del manual
- Usa emojis relevantes (🔧⚡🏠💡🚰🖼️) para hacer la respuesta más clara
- Si hay pasos específicos, númeralos claramente
- Si hay advertencias importantes, márcalas como ⚠️ IMPORTANTE
- Si hay material visual disponible, menciona que existen imágenes técnicas para complementar la explicación
- Mantén un tono profesional pero amigable
- Si no tienes información suficiente en el contexto, dilo claramente

**RESPUESTA:**"""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-70b-versatile",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Eres un experto en mantenimiento de edificios religiosos, especializado en Salones del Reino. Siempre basas tus respuestas en el manual oficial e incluyes referencias a material visual cuando está disponible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    logger.info("✅ Respuesta generada con Groq")
                    
                    # Retornar respuesta con imágenes
                    return {
                        "text": answer,
                        "images": relevant_images
                    }
                else:
                    logger.error(f"❌ Error Groq: {response.status_code}")
                    return self._format_mock_answer_with_images(context_chunks[0])
                    
        except Exception as e:
            logger.error(f"❌ Error generando respuesta: {str(e)}")
            return self._format_mock_answer_with_images(context_chunks[0] if context_chunks else {})
    
    def _format_mock_answer_with_images(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Formatear respuesta mock incluyendo imágenes"""
        if not chunk:
            return {
                "text": "❌ No encontré información específica sobre tu consulta en el manual.",
                "images": []
            }
        
        content = chunk.get('content', 'Información no disponible')
        page = chunk.get('page', 0)
        title = chunk.get('title', 'Manual de Mantenimiento')
        images = chunk.get('images', [])
        
        # Agregar nota sobre imágenes si están disponibles
        image_note = f"\n\n🖼️ *{len(images)} imagen(es) técnica(s) disponible(s)*" if images else ""
        
        text_response = f"""🔧 **{title}** (Página {page})

{content}{image_note}

---
💡 *Respuesta del Manual de Mantenimiento de Salones del Reino*
⚠️ *Para consultas complejas, contacta al formador de mantenimiento*"""

        return {
            "text": text_response,
            "images": images
        }
    
    async def query(self, user_query: str, user_id: str = "default") -> Dict[str, Any]:
        """Método principal para procesar consultas con soporte para imágenes"""
        try:
            logger.info(f"🔍 Procesando consulta: {user_query[:50]}... (usuario: {user_id})")
            
            # 1. Generar embedding de la consulta usando API remota
            query_embedding = await self.get_remote_embedding(user_query)
            
            # 2. Buscar chunks relevantes en Qdrant
            relevant_chunks = await self.search_qdrant_vectors(query_embedding, limit=3)
            
            # 3. Generar respuesta con imágenes usando Groq
            answer_data = await self.generate_answer_with_groq(user_query, relevant_chunks)
            
            # 4. Preparar sources
            sources = []
            for chunk in relevant_chunks:
                source = f"Manual - Página {chunk['page']}: {chunk['title']}"
                if chunk.get('score', 0) > 0:
                    source += f" (relevancia: {chunk['score']:.2f})"
                if chunk.get('has_images'):
                    source += f" [📸 {chunk.get('image_count', 0)} imagen(es)]"
                sources.append(source)
            
            # 5. Contar estadísticas de imágenes
            total_images = len(answer_data["images"])
            if total_images > 0:
                logger.info(f"🖼️ Respuesta incluye {total_images} imagen(es)")
            
            return {
                "answer": answer_data["text"],
                "images": answer_data["images"],  # URLs de imágenes relevantes
                "sources": sources,
                "metadata": {
                    "query": user_query,
                    "user_id": user_id,
                    "chunks_found": len(relevant_chunks),
                    "images_found": total_images,
                    "system_status": "remote_embeddings_with_images" if self.operational else "limited_mock",
                    "embedding_method": "huggingface_remote",
                    "search_method": "qdrant_vector_search",
                    "features": ["text_search", "image_support", "remote_embeddings"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en consulta: {str(e)}")
            return {
                "answer": f"❌ Error procesando consulta: {str(e)}",
                "images": [],
                "sources": [],
                "metadata": {
                    "error": str(e), 
                    "user_id": user_id,
                    "system_status": "error"
                }
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar estado del sistema"""
        status = {
            "qdrant_configured": bool(self.qdrant_url and self.qdrant_api_key),
            "groq_configured": bool(self.groq_api_key),
            "embedding_method": "huggingface_remote",
            "image_support": True,
            "operational": self.operational,
            "overall_status": "healthy" if self.operational else "limited",
            "features": {
                "vector_search": True,
                "remote_embeddings": True,
                "image_extraction": True,
                "image_serving": True,
                "groq_llm": True
            }
        }
        
        return status
