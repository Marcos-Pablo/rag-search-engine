from sentence_transformers import SentenceTransformer
import numpy as np
import os, re, json

from lib.search_utils import (
    CACHE_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.movie_embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

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
        with open(self.movie_embeddings_path, "wb") as file:
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

    def search(self, query: str, limit: int):
        if self.documents is None or self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        scores: list[tuple[float, dict]] = []

        for doc, doc_embedding in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            scores.append((similarity, doc))

        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in sorted_scores[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_PATH, "chunk_metadata.json")

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            with open(self.chunk_embeddings_path, "rb") as file:
                self.chunk_embeddings = np.load(file)

            with open(self.chunk_metadata_path, "r") as file:
                data = json.load(file)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text:
                continue
            self.document_map[doc["id"]] = doc
            chunks = semantic_chunk(
                text, DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
            )
            for i, curr_chunk in enumerate(chunks):
                all_chunks.append(curr_chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        with open(self.chunk_embeddings_path, "wb") as file:
            np.save(file, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as file:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                file,
                indent=2,
            )
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_scores_list = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )

        top_results = []
        for movie_idx, score in sorted_scores_list[:limit]:
            doc = self.documents[movie_idx]
            top_results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    "score": round(score, SCORE_PRECISION),
                }
            )

        return top_results


def search_chunked_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    chunked_sem_search = ChunkedSemanticSearch()
    chunked_sem_search.load_or_create_chunk_embeddings(movies)
    result = chunked_sem_search.search_chunks(query, limit)
    for i, record in enumerate(result):
        print(f"\n{i}. {record['title']} (score: {record['score']:.4f})")
        print(f"   {record['document']}...")


def embed_chunks_command():
    movies = load_movies()
    chunked_sem_search = ChunkedSemanticSearch()
    embeddings = chunked_sem_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    sem_search = SemanticSearch()
    movies = load_movies()
    sem_search.load_or_create_embeddings(movies)
    results = sem_search.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        return [text]

    chunk_size = max(1, max_chunk_size)
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i : i + chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int):
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def fixed_size_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    overlap = max(overlap, 0)
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def chunk_command(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = fixed_size_chunk(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


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
