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

def evaluate_model_performance(model_name, test_sentences):
    model = SentenceTransformer(model_name)
    start_time = time.time()
    embeddings = model.encode(test_sentences)
    encode_time = time.time() - start_time

    similarities = []
    for i in range(0, len(embeddings), 2):  # Paare vergleichen
        if i + 1 < len(embeddings):
            sim = 1 - spatial.distance.cosine(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

    return {
        'model': model_name,
        'encoding_time': encode_time,
        'avg_similarity': np.mean(similarities) if similarities else None,
        'embedding_dim': embeddings.shape[1]
    }

# Modelle zum Vergleich
model_names = [
    'distiluse-base-multilingual-cased-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'paraphrase-multilingual-mpnet-base-v2',
    'aari1995/German_Semantic_STS_V2'
]

results = {}

for model_name in model_names:
    print(f"\n\033[93m=== Ergebnisse für Modell: {model_name} ===\033[0m")
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(german_documents, convert_to_tensor=True)
    query_embeddings = model.encode(german_queries, convert_to_tensor=True)
    for i, query in enumerate(german_queries):
        similarities = util.cos_sim(query_embeddings[i], doc_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        interpretation = interpret_similarity(best_score)
        print(f"Query: {query}")
        print(f"Relevantestes Dokument: {german_documents[best_idx]}")
        print(f"Score: {best_score:.4f} - {interpretation}\n")

    # Performance-Metriken ausgeben
    perf = evaluate_model_performance(model_name, german_documents + german_queries)
    print(f"Performance: Encoding-Zeit={perf['encoding_time']:.2f}s, "
          f"Durchschnittliche Similarity={perf['avg_similarity']:.4f}, "
          f"Embedding-Dimension={perf['embedding_dim']}")