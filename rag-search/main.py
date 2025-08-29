"""
RAG-System mit Ollama in Docker und lokaler Python App
Ollama läuft in Docker, RAG App lokal
"""

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import time
import os
from typing import List, Dict

class LocalRAG:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """RAG System mit Docker Ollama"""
        self.ollama_url = ollama_url
        print(f"Ollama URL: {self.ollama_url}")
        
        # Embedding Model laden (lokal)
        print("Lade Embedding Model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ChromaDB (lokal, persistent)
        chroma_path = "./chroma_db"
        os.makedirs(chroma_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.client.delete_collection("products")
        except:
            pass
        self.collection = self.client.create_collection(
            name="products",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Ollama Model
        self.model_name = "llama3.2:3b"
        
        print("LocalRAG System bereit!")
    
    def check_ollama(self) -> bool:
        """Prüft ob Ollama Docker Container läuft"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_ollama(self, timeout: int = 60) -> bool:
        """Wartet bis Ollama Container bereit ist"""
        print("Warte auf Ollama Docker Container...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_ollama():
                print("Ollama Container ist bereit!")
                return True
            time.sleep(2)
            print("   ...")
        
        print("Ollama Container nicht erreichbar!")
        print("Starte Ollama mit: docker-compose up -d")
        return False
    
    def pull_model(self, model_name: str = None) -> bool:
        """Lädt Llama Model herunter"""
        if not model_name:
            model_name = self.model_name
            
        print(f"Lade Model {model_name} herunter...")
        print("Das kann beim ersten Mal einige Minuten dauern...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        print(f"   {data['status']}")
                    if data.get("status") == "success":
                        print("Model erfolgreich geladen!")
                        return True
            return False
        except Exception as e:
            print(f"Fehler beim Model-Download: {e}")
            return False
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Dokumente in kleinere Chunks aufteilen"""
        chunks = []
        for doc in documents:
            sentences = doc['content'].split('. ')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunks.append({
                        'id': f"{doc['id']}_chunk_{i}",
                        'content': sentence.strip() + '.',
                        'source': doc['id'],
                        'metadata': doc.get('metadata', {})
                    })
        return chunks
    
    def add_documents(self, documents: List[Dict]):
        """Dokumente zur Vector Database hinzufügen"""
        print("Verarbeite Dokumente...")
        chunks = self.chunk_documents(documents)
        
        # Embeddings erstellen
        contents = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True).tolist()
        
        # Zu ChromaDB hinzufügen
        self.collection.add(
            ids=[chunk['id'] for chunk in chunks],
            embeddings=embeddings,
            documents=contents,
            metadatas=[{'source': chunk['source'], **chunk['metadata']} for chunk in chunks]
        )
        
        print(f"{len(chunks)} Chunks zur Datenbank hinzugefügt")
    
    def similarity_search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Semantic Search in der Vector Database"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return [{
            'content': doc,
            'distance': dist,
            'metadata': meta
        } for doc, dist, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        )]
    
    def generate_answer_local(self, query: str, context_docs: List[Dict]) -> str:
        """Antwort mit lokalem Llama generieren"""
        context = "\n".join([f"- {doc['content']}" for doc in context_docs])
        
        # Optimiertes Prompt für Llama
        prompt = f"""Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext.

Kontext:
{context}

Frage: {query} """

        try:
            print("Generiere Antwort mit Llama...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 200
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Keine Antwort erhalten').strip()
            else:
                return f"Fehler bei LLM-Anfrage: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Timeout: LLM-Antwort dauerte zu lange"
        except Exception as e:
            return f"Fehler bei LLM-Kommunikation: {str(e)}"
    
    def query(self, question: str) -> Dict:
        """RAG Pipeline ausführen"""
        print(f"\nBearbeite Frage: {question}")
        
        # 1. Similarity Search
        start_time = time.time()
        relevant_docs = self.similarity_search(question, n_results=3)
        search_time = time.time() - start_time
        
        print(f"    Suche: {search_time:.2f}s, {len(relevant_docs)} Dokumente gefunden")
        
        # 2. LLM Antwort generieren
        start_time = time.time()
        answer = self.generate_answer_local(question, relevant_docs)
        llm_time = time.time() - start_time
        
        print(f"    LLM: {llm_time:.2f}s")
        
        return {
            'question': question,
            'answer': answer,
            'sources': relevant_docs,
            'timing': {
                'search': search_time,
                'llm': llm_time
            }
        }

def loadConentFromFile():
    if os.path.exists("data.json"):
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return PRODUCT_CATALOG

PRODUCT_CATALOG = loadConentFromFile()


def main():
    """Hauptfunktion - Setup und Demo"""
    print("🚀 RAG System mit Docker Ollama")
    print("=" * 40)
    
    # RAG System initialisieren
    rag = LocalRAG()
    
    # 1. Ollama Docker Container prüfen
    print("\n1️⃣ Prüfe Ollama Docker Container...")
    if not rag.wait_for_ollama():
        print("\nOllama starten:")
        print("   docker run -d -p 11434:11434 --name ollama ollama/ollama")
        return False
    
    # 3. Dokumente hinzufügen
    print("\n3️⃣ Lade Demo-Daten...")
    rag.add_documents(PRODUCT_CATALOG)
    
    # 4. Interactive Demo
    print("\n4️⃣ Interactive RAG Demo")
    print("=" * 25)
    print("Gib 'quit' ein zum Beenden\n")
    
    while True:
        try:
            question = input("Deine Frage: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Auf Wiedersehen!")
                break
            
            if not question:
                continue
            
            print("\nSuche nach Antwort...")
            result = rag.query(question)
            
            print(f"\nAntwort:")
            print(f"   {result['answer']}")
            print(f"\nStats: Suche {result['timing']['search']:.2f}s, LLM {result['timing']['llm']:.2f}s")
            print(f"Quellen: {len(result['sources'])} Dokumente")
            ### Details zu Quellen anzeigen
            for src in result['sources']:
                print(f" - {src['metadata'].get('source', 'unknown')} (Dist: {src['distance']:.4f})")
                print(f"   {src['content']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nAuf Wiedersehen!")
            break
        except Exception as e:
            print(f"Fehler: {e}")

if __name__ == "__main__":
    main()
