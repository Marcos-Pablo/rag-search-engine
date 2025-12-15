from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

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
