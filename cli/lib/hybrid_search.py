import os

from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_WEIGHTED_SEARCH_ALPHA, load_movies
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit=5):
        bm25_result = self._bm25_search(query, limit * 500)
        semantic_result = self.semantic_search.search_chunks(query, limit * 500)
        bm25_normalized = normalize_scores([item["score"] for item in bm25_result])
        semantic_normalized = normalize_scores([item["score"] for item in semantic_result])
        combined_scores = {}

        for i, item in enumerate(bm25_result):
            doc_id = item["id"]
            if doc_id not in combined_scores:
                combined_scores[item["id"]] = {
                    "title": self.idx.docmap[item["id"]]["title"],
                    "description": self.idx.docmap[item["id"]]["description"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }

            if bm25_normalized[i] > combined_scores[doc_id]["bm25_score"]:
                combined_scores[doc_id]["bm25_score"] = bm25_normalized[i]

        for i, item in enumerate(semantic_result):
            doc_id = item["id"]
            if doc_id not in combined_scores:
                combined_scores[item["id"]] = {
                    "title": self.semantic_search.document_map[item["id"]]["title"],
                    "description": self.idx.docmap[item["id"]]["description"],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                }

            if semantic_normalized[i] > combined_scores[doc_id]["semantic_score"]:
                combined_scores[doc_id]["semantic_score"] = semantic_normalized[i]

        for item in combined_scores.values():
            item["hybrid_score"] = hybrid_score(item["bm25_score"], item["semantic_score"], alpha)

        top_results = sorted(list(combined_scores.values()), key=lambda item: item["hybrid_score"], reverse=True)
        return top_results[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha=DEFAULT_WEIGHTED_SEARCH_ALPHA, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    result = hybrid_search.weighted_search(query, alpha, limit)

    print(
        f"Weighted Hybrid Search Results for '{query}' (alpha={alpha}):"
    )
    print(
        f"  Alpha {alpha}: {int(alpha * 100)}% Keyword, {int((1 - alpha) * 100)}% Semantic"
    )
    for i, res in enumerate(result, 1):
        print(f"{i}. {res['title']}")
        print(f"   Hybrid Score: {res.get('hybrid_score', 0):.3f}")
        print(
            f"   BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}"
        )
        print(f"   {res['description'][:100]}...")
        print()

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score, max_score = float("inf"), float("-inf")
    for score in scores:
        min_score = min(min_score, score)
        max_score = max(max_score, score)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_scores.append(normalized_score)
    return normalized_scores

def normalize_scores_command(scores: list[float]):
    normalized_scores = normalize_scores(scores)
    for score in normalized_scores:
        print(f"* {score:.4f}")
