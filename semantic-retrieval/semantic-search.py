from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
from scipy import spatial

german_documents = [
    "Python ist eine beliebte Programmiersprache fuer Data Science",
    "Machine Learning Algorithmen analysieren grosse Datenmengen",
    "Kuenstliche Intelligenz revolutioniert viele Branchen",
    "Das Wetter heute ist sonnig und warm",
    "Berlin ist die Hauptstadt von Deutschland"
]

german_queries = [
    "Welche Sprache nutzt man fuer Datenanalyse?",
    "Wie funktioniert maschinelles Lernen?",
    "Was ist KI?",
    "Wie ist das Wetter?",
    "Hauptstadt Deutschland"
]

def interpret_similarity(score):
    if score > 0.8:
        return "Sehr aehnlich (duplicate/paraphrase)"
    elif score > 0.6:
        return "Aehnlich (related topic)"
    elif score > 0.4:
        return "Etwas aehnlich (weak relation)"
    else:
        return "Unaehnlich (different topics)"

# Modelle zum Vergleich
model_names = [
    'distiluse-base-multilingual-cased-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'paraphrase-multilingual-mpnet-base-v2',
    'aari1995/German_Semantic_STS_V2',
    'Sahajtomar/German-semantic'
]

results = {}

for model_name in model_names:
    print(f"\n\033[93m=== Ergebnisse für Modell: {model_name} ===\033[0m")
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(german_documents, convert_to_tensor=True)
    query_embeddings = model.encode(german_queries, convert_to_tensor=True)
    best_scores = []
    for i, query in enumerate(german_queries):
        similarities = util.cos_sim(query_embeddings[i], doc_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_scores.append(best_score)
        interpretation = interpret_similarity(best_score)
        print(f"Query: {query}")
        print(f"Relevantestes Dokument: {german_documents[best_idx]}")
        print(f"Score: {best_score:.4f} - {interpretation}\n")

    avg_best_score = np.mean(best_scores)
    print(f"🔹 Durchschnittlicher Best-Score für {model_name}: {avg_best_score:.4f}")
