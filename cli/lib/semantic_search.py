from sentence_transformers import SentenceTransformer
import numpy as np
import os

from lib.search_utils import CACHE_PATH, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.movie_embeddings_path = os.path.join(CACHE_PATH, 'movie_embeddings.npy')

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        with open(self.movie_embeddings_path, 'wb') as file:
            np.save(file, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.movie_embeddings_path):
            self.embeddings = np.load(self.movie_embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    sem_search = SemanticSearch()
    documents = load_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text_command(text):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model_command():
    sm_search = SemanticSearch()
    print(f"Model loaded: {sm_search.model}")
    print(f"Max sequence length: {sm_search.model.max_seq_length}")
