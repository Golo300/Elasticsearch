from sentence_transformers import SentenceTransformer, util

german_documents = [
    "Python ist eine beliebte Programmiersprache für Data Science",
    "Machine Learning Algorithmen analysieren grosse Datenmengen",
    "Künstliche Intelligenz revolutioniert viele Branchen",
    "Das Wetter heute ist sonnig und warm",
    "Berlin ist die Hauptstadt von Deutschland"
]

german_queries = [
    "Welche Sprache nutzt man für Datenanalyse?",
    "Wie funktioniert maschinelles Lernen?",
    "Was ist KI?",
    "Wie ist das Wetter?",
    "Hauptstadt Deutschland"
]

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

doc_embeddings = model.encode(german_documents, convert_to_tensor=True)
query_embeddings = model.encode(german_queries, convert_to_tensor=True)

for query, query_emb in zip(german_queries, query_embeddings):
    scores = util.cos_sim(query_emb, doc_embeddings)[0]
    best_idx = scores.argmax().item()
    print(f"\nQuery: {query}")
    print(f"Am ähnlichsten: {german_documents[best_idx]} (Score: {scores[best_idx]:.4f})")

