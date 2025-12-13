from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

def verify_model_command():
    sm_search = SemanticSearch()
    print(f"Model loaded: {sm_search.model}")
    print(f"Max sequence length: {sm_search.model.max_seq_length}")
