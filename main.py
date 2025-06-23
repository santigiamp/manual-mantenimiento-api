from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import groq
import os
from typing import List, Dict

app = FastAPI(title="Manual Mantenimiento API")

# Configuración
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar componentes
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = groq.Groq(api_key=GROQ_API_KEY)

class ChatRequest(BaseModel):
    query: str
    user_id: str
    context: str = "telegram"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Generar embedding
        query_embedding = embedding_model.encode(request.query).tolist()
        
        # Buscar en Qdrant
        results = qdrant_client.search(
            collection_name="manual_mantenimiento",
            query_vector=query_embedding,
            limit=3
        )
        
        if not results:
            return ChatResponse(
                answer="❌ No encontré información relevante sobre tu consulta.",
                sources=[],
                confidence=0.0
            )
        
        # Preparar contexto para Groq
        context = "\n\n".join([
            f"Fuente {i+1} (Página {r.payload['page']}):\n{r.payload['text'][:500]}"
            for i, r in enumerate(results)
        ])
        
        # Generar respuesta con Groq
        prompt = f"""Eres un experto en mantenimiento. Responde basándote únicamente en el contexto del manual.

CONTEXTO:
{context}

PREGUNTA: {request.query}

RESPUESTA (sé específico y práctico):"""
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un experto en mantenimiento de edificios."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        # Formatear fuentes
        sources = [
            {
                "section": r.payload.get('section', 'Sin sección'),
                "page": r.payload['page'],
                "text": r.payload['text'][:200] + "...",
                "score": r.score
            }
            for r in results
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=results[0].score if results else 0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Manual Mantenimiento API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
