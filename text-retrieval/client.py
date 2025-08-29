from elasticsearch import Elasticsearch
import os

class TextSearcher:
    def __init__(self, host='localhost', port=9200):
        """
        Initialisiert die Elasticsearch-Verbindung
        """
        self.es = Elasticsearch([f'http://{host}:{port}'])
        self.index_name = 'text_documents'
        
    def create_index(self):
        """
        Erstellt einen Index für die Textdokumente
        """
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "german"
                    },
                    "filename": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)
            print(f"Index '{self.index_name}' wurde erstellt.")
        else:
            print(f"Index '{self.index_name}' existiert bereits.")
    
    def index_text_file(self, file_path='./data/stupro_list.txt'):

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Dokument in Elasticsearch indexieren
            doc = {
                'content': content,
                'filename': file_path
            }
            
            response = self.es.index(
                index=self.index_name,
                id=1,  # Einfache ID für das Dokument
                body=doc
            )
            
            print(f"Datei '{file_path}' wurde erfolgreich indexiert.")
            return response
            
        except FileNotFoundError:
            print(f"Datei '{file_path}' nicht gefunden!")
            return None
        except Exception as e:
            print(f"Fehler beim Indexieren: {e}")
            return None
    
    def search_text(self, query):

        search_body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "fuzziness": "AUTO"
                    }
                }
            },
            # "query": { # funktioniert nicht so richtig mit AND und OR
            #     "simple_query_string": {
            #         "query": query,  # z.B. "Bachelor AND DHBW"
            #         "fields": ["content"],
            #         "default_operator": "and",
            #     }
            # },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 10,
                        "pre_tags": ["\u001b[33m"], # Gelb
                        "post_tags": ["\u001b[0m"],
                        "order": "score"
                    }
                }
            }
        }
        
        try:
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            return self.format_results(response, query)
            
        except Exception as e:
            print(f"Fehler bei der Suche: {e}")
            return None
    
    def format_results(self, response, query):
        """
        Formatiert die Suchergebnisse für die Ausgabe
        """
        hits = response['hits']['hits']
        
        print(f"\n=== Suchergebnisse für '{query}' ===")
        print(f"Gefunden: {len(hits)} Treffer\n")
        
        if hits:
            for hit in hits:
                score = hit['_score']
                print(f"Relevanz-Score: {score:.2f}")
                
                if 'highlight' in hit:
                    highlights = hit['highlight']['content']
                    for i, highlight in enumerate(highlights, 1):
                        print(f"{i}: {highlight}")
                        print("\n")
                
                print("-" * 50)
        else:
            print("Keine Treffer gefunden.")
        
        return hits

def main():

    searcher = TextSearcher()
    
    try:
        if not searcher.es.ping():
            print("Fehler: Elasticsearch ist nicht erreichbar!")
            print("Stelle sicher, dass Elasticsearch läuft (normalerweise auf http://localhost:9200)")
            return
            
    except Exception as e:
        print(f"Verbindungsfehler zu Elasticsearch: {e}")
        return
    
    searcher.create_index()
    
    file_path = './data/stupro_list.txt'
    print(f"Indexiere {file_path}...")
    searcher.index_text_file(file_path)
    
    print("\n" + "="*50)
    print("Elasticsearch Textsuche gestartet!")
    print("Geben Sie 'quit' ein, um das Programm zu beenden.")
    print("="*50)
    
    while True:
        query = input("\nSuchbegriff eingeben: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Programm beendet.")
            break
        
        if query:
            searcher.search_text(query)
        else:
            print("Bitte geben Sie einen Suchbegriff ein.")

if __name__ == "__main__":
    main()