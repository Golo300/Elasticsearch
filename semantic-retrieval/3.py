from sentence_transformers import SentenceTransformer
from scipy import spatial

import numpy as dnp

# Download model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#sprachunabhängig
#model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# Modelle siehe https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

# The sentences we'd like to encode
sentences = ['Python is an interpreted high-level general-purpose programming language.',
    'Python is dynamically-typed and garbage-collected.',
    'Python is a great general-purpose programming language',
    'The quick brown fox jumps over the lazy dog.']

# Get embeddings of sentences
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
# Sentence: Python is an interpreted high-level general-purpose programming language.
# Embedding: [-1.17965914e-01 -4.57159936e-01 -5.87313235e-01 -2.72477478e-01 ...
# ...

# Alle Kombinationen berechnen
for i in range(0,len(embeddings)):
    print("From ", sentences[i])
    for j in range(0,i+1):
        print("to ", sentences[j], " ",1 - spatial.distance.cosine(embeddings[i], embeddings[j]))
    print()
