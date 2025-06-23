from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
import groq
import os
from typing import List, Dict
import re

app = FastAPI(title="Manual Mantenimiento API - Lite")

# Configuración
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("🔧 Inicializando API...")

# Clientes ligeros
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)

print("✅ API lista")

class ChatRequest(BaseModel):
    query: str
    user_id: str
    context: str = "telegram"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float

def extract_keywords(query: str) -> List[str]:
    """Extraer keywords de la consulta"""
    # Limpiar y extraer palabras importantes
    keywords = re.findall(r'\b\w{4,}\b', query.lower())
    
    # Keywords específicos del manual
    important_terms = [
        'mantenimiento', 'aire', 'acondicionado', 'extintor', 
        'eléctrico', 'bomba', 'agua', 'grietas', 'sistema',
        'inspección', 'reparar', 'limpiar', 'revisar'
    ]
    
    # Priorizar términos importantes
    final_keywords = []
    for term in important_terms:
        if term in query.lower():
            final_keywords.append(term)
    
    # Agregar otras keywords
    for keyword in keywords:
        if len(keyword) > 3 and keyword not in final_keywords:
            final_keywords.append(keyword)
    
    return final_keywords[:5]  # Máximo 5 keywords

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"🔍 Query: {request.query}")
        
        # Extraer keywords
        keywords = extract_keywords(request.query)
        print(f"📋 Keywords: {keywords}")
        
        # Buscar en Qdrant usando scroll (más eficiente que search sin embeddings)
        all_results = []
        
        # Buscar por cada keyword
        for keyword in keywords:
            try:
                results, _ = qdrant_client.scroll(
                    collection_name="manual_mantenimiento",
                    scroll_filter={
                        "must": [
                            {
                                "key": "text",
                                "match": {
                                    "text": keyword
                                }
                            }
                        ]
                    },
                    limit=5
                )
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Error buscando '{keyword}': {e}")
                continue
        
        # Si no hay resultados, buscar sin filtros
        if not all_results:
            try:
                all_results, _ = qdrant_client.scroll(
                    collection_name="manual_mantenimiento",
                    limit=10
                )
                print("📝 Búsqueda sin filtros")
            except Exception as e:
                print(f"❌ Error en búsqueda general: {e}")
        
        if not all_results:
            return ChatResponse(
                answer="❌ No encontré información relevante en el manual. Intenta reformular tu consulta.",
                sources=[],
                confidence=0.0
            )
        
        # Tomar mejores resultados (eliminar duplicados)
        unique_results = {}
        for result in all_results:
            result_id = result.payload.get('text', '')[:100]
            if result_id not in unique_results:
                unique_results[result_id] = result
        
        best_results = list(unique_results.values())[:3]
        
        # Preparar contexto para Groq
        context_parts = []
        for i, result in enumerate(best_results):
            text = result.payload.get('text', '')
            section = result.payload.get('section', 'Manual')
            page = result.payload.get('page', 0)
            
            context_parts.append(f"""
FUENTE {i+1} - {section} (Página {page}):
{text[:600]}
""")
        
        context = "\n".join(context_parts)
        
        # Generar respuesta con Groq
        prompt = f"""Eres un experto en mantenimiento de edificios. Responde basándote únicamente en la información del manual proporcionado.

INFORMACIÓN DEL MANUAL:
{context}

PREGUNTA DEL USUARIO: {request.query}

INSTRUCCIONES:
- Responde de manera práctica y específica
- Si es un procedimiento, menciona los pasos
- Incluye números de página cuando sea relevante
- Si no hay información suficiente, dilo claramente
- Usa un tono profesional pero accesible

RESPUESTA:"""
        
        print("🤖 Generando respuesta con Groq...")
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un experto técnico en mantenimiento de edificios."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        # Formatear fuentes
        sources = []
        for result in best_results:
            sources.append({
                "section": result.payload.get('section', 'Manual de Mantenimiento'),
                "page": result.payload.get('page', 0),
                "text": result.payload.get('text', '')[:200] + "...",
                "score": 0.8  # Score fijo ya que no usamos embeddings
            })
        
        print("✅ Respuesta generada")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=0.8 if best_results else 0.0
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Manual Mantenimiento API - Lite",
        "qdrant_connected": bool(qdrant_client),
        "groq_connected": bool(groq_client)
    }

@app.get("/")
async def root():
    return {"message": "Manual Mantenimiento API funcionando", "version": "lite"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
