import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq
import uuid

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        """Inicializar el sistema RAG con todas las conexiones"""
        try:
            # Configuración desde variables de entorno
            self.qdrant_url = os.getenv("QDRANT_URL")
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY") 
            self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "manual_mantenimiento")
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            
            # Validar configuración
            if not all([self.qdrant_url, self.qdrant_api_key, self.groq_api_key]):
                raise ValueError("Faltan variables de entorno: QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY")
            
            # Inicializar clientes
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
            
            self.groq_client = Groq(api_key=self.groq_api_key)
            
            # Modelo de embeddings (compatible con CPU)
            logger.info("Cargando modelo de embeddings...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sistema RAG inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando RAG: {str(e)}")
            self.qdrant_client = None
            self.groq_client = None
            self.embedding_model = None
    
    def create_collection_if_not_exists(self):
        """Crear la colección si no existe"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creando colección: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Dimensión del modelo all-MiniLM-L6-v2
                        distance=Distance.COSINE
                    )
                )
                logger.info("Colección creada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error creando colección: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Agregar documentos a la base vectorial"""
        try:
            if not self.qdrant_client or not self.embedding_model:
                raise ValueError("Sistema RAG no inicializado correctamente")
            
            # Crear colección si no existe
            self.create_collection_if_not_exists()
            
            points = []
            for i, doc in enumerate(documents):
                # Generar embedding
                text = doc.get('content', '')
                embedding = self.embedding_model.encode(text).tolist()
                
                # Crear punto
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'content': text,
                        'page': doc.get('page', 0),
                        'section': doc.get('section', ''),
                        'title': doc.get('title', ''),
                        'chunk_id': i
                    }
                )
                points.append(point)
            
            # Insertar en Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Agregados {len(documents)} documentos a la base vectorial")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando documentos: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Buscar chunks similares en la base vectorial"""
        try:
            if not self.qdrant_client or not self.embedding_model:
                logger.warning("Sistema RAG no disponible, usando datos mock")
                return self._get_mock_results(query)
            
            # Generar embedding de la consulta
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Buscar en Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )
            
            # Formatear resultados
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload.get('content', ''),
                    'page': result.payload.get('page', 0),
                    'section': result.payload.get('section', ''),
                    'title': result.payload.get('title', ''),
                    'score': result.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {str(e)}")
            return self._get_mock_results(query)
    
    def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Resultados mock cuando no hay conexión a Qdrant"""
        query_lower = query.lower()
        
        if "aire acondicionado" in query_lower or "gotea" in query_lower:
            return [{
                'content': """Aire acondicionado: Gotea la unidad interior
                
Identificar la punta de la manguera de desagüe del equipo y verificar que no esté 
obstruida en la salida. Si el problema no está allí introducir una cinta pasacable 
a través de la manguera hasta lograr desobstruirla. En algún punto de este proceso 
debería comenzar a salir agua por la manguera.""",
                'page': 43,
                'section': 'G',
                'title': 'Reparaciones Rápidas - Aire Acondicionado',
                'score': 0.95
            }]
        
        elif "pintura" in query_lower or "pared" in query_lower:
            return [{
                'content': """Paredes: Pintura dañada
                
Para reparar marcas, raspones, rayones y manchas:
1. Limpiar la superficie con cepillo o trapo
2. Restaurar rayones y marcas con masilla o enduido  
3. Lijar dejando la superficie alisada y uniforme
4. Eliminar todo el polvo y aplicar fijador/sellador
5. Pintar con pincel o rodillo según el tamaño de la superficie""",
                'page': 42,
                'section': 'E',
                'title': 'Reparaciones Rápidas - Pintura',
                'score': 0.92
            }]
        
        elif "grieta" in query_lower or "rajadura" in query_lower:
            return [{
                'content': """Muro de ladrillos: Rajaduras y grietas
                
Para grietas pequeñas y fisuras que sólo afecten los revoques o la pintura:
1. Introducir el canto de una espátula en la grieta y abrirla en forma de "V"
2. Con un pincel seco dejar el interior completamente libre de polvo
3. Rellenar con material apropiado (masilla plástica interior o exterior)
4. Dejar secar y lijar hasta nivelar la pared
                
IMPORTANTE: Si las grietas son más profundas y afectan la estructura, 
comunicarse con el formador de mantenimiento.""",
                'page': 40,
                'section': 'B', 
                'title': 'Reparaciones Rápidas - Grietas',
                'score': 0.89
            }]
        
        else:
            return [{
                'content': """Manual de Mantenimiento - Salones del Reino
                
Este manual contiene información sobre mantenimiento preventivo y correctivo 
para edificios, sistemas eléctricos, sistemas electrónicos, sistemas mecánicos 
y reparaciones rápidas.""",
                'page': 1,
                'section': 'Introducción',
                'title': 'Manual de Mantenimiento',
                'score': 0.7
            }]
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generar respuesta usando Groq"""
        try:
            if not self.groq_client:
                return self._generate_mock_answer(query, context_chunks)
            
            # Preparar contexto
            context = "\n\n".join([
                f"**{chunk['title']} (Página {chunk['page']})**\n{chunk['content']}"
                for chunk in context_chunks
            ])
            
            # Prompt para el LLM
            prompt = f"""Eres un asistente experto en mantenimiento de Salones del Reino. 
Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto del manual proporcionado.

**CONTEXTO DEL MANUAL:**
{context}

**PREGUNTA DEL USUARIO:**
{query}

**INSTRUCCIONES:**
- Responde en español con información práctica y específica
- Usa emojis relevantes para hacer la respuesta más clara
- Si hay pasos específicos, númeralos 
- Si hay advertencias importantes, márcalas como ⚠️ IMPORTANTE
- Si no tienes información suficiente en el contexto, dilo claramente
- Mantén un tono profesional pero amigable

**RESPUESTA:**"""

            # Llamar a Groq
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "Eres un experto en mantenimiento de edificios religiosos, especializado en Salones del Reino."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            return self._generate_mock_answer(query, context_chunks)
    
    def _generate_mock_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Respuesta mock cuando Groq no está disponible"""
        if not context_chunks:
            return "❌ No encontré información específica sobre tu consulta en el manual."
        
        chunk = context_chunks[0]
        return f"""🔧 **{chunk['title']}** (Página {chunk['page']})

{chunk['content']}

---
💡 *Respuesta generada desde el Manual de Mantenimiento de Salones del Reino*
⚠️ *Sistema en modo desarrollo - Para consultas complejas, contacta al formador de mantenimiento*"""
    
    def query(self, user_query: str, user_id: str = "default") -> Dict[str, Any]:
        """Método principal para procesar consultas"""
        try:
            logger.info(f"Procesando consulta: {user_query[:50]}...")
            
            # 1. Buscar chunks relevantes
            relevant_chunks = self.search_similar_chunks(user_query, limit=3)
            
            # 2. Generar respuesta
            answer = self.generate_answer(user_query, relevant_chunks)
            
            # 3. Preparar sources
            sources = [
                f"Manual - Página {chunk['page']}: {chunk['title']}"
                for chunk in relevant_chunks
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "query": user_query,
                    "user_id": user_id,
                    "chunks_found": len(relevant_chunks),
                    "system_status": "operational" if self.qdrant_client and self.groq_client else "limited"
                }
            }
            
        except Exception as e:
            logger.error(f"Error en consulta: {str(e)}")
            return {
                "answer": f"❌ Error procesando consulta: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e), "user_id": user_id}
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar estado del sistema"""
        status = {
            "qdrant_connected": False,
            "groq_connected": False,
            "embedding_model_loaded": False,
            "overall_status": "error"
        }
        
        try:
            # Test Qdrant
            if self.qdrant_client:
                collections = self.qdrant_client.get_collections()
                status["qdrant_connected"] = True
                status["collections"] = [col.name for col in collections.collections]
            
            # Test Groq
            if self.groq_client:
                status["groq_connected"] = True
            
            # Test Embedding Model
            if self.embedding_model:
                status["embedding_model_loaded"] = True
            
            # Overall status
            if all([status["qdrant_connected"], status["groq_connected"], status["embedding_model_loaded"]]):
                status["overall_status"] = "healthy"
            elif any([status["qdrant_connected"], status["groq_connected"]]):
                status["overall_status"] = "partial"
            
            return status
            
        except Exception as e:
            status["error"] = str(e)
            return status
