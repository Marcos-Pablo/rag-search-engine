import os

from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies
from lib.semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents=[], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])  # type: ignore[arg-type]
        return image_embedding[0]

    def search_with_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img_embedding = self.embed_image(image_path)
        results = []
        for txt_embedding, doc in zip(self.text_embeddings, self.documents):
            similarity = cosine_similarity(img_embedding, txt_embedding)
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity_score": similarity,
                }
            )

        results.sort(key=lambda item: item["similarity_score"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path: str):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    movies = load_movies()
    searcher = MultimodalSearch(movies)
    return searcher.search_with_image(image_path)
