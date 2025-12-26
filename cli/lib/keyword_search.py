import string, pickle, os, math
from typing import Counter, DefaultDict
from nltk.stem import PorterStemmer
from nltk.util import defaultdict
from lib.search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    SCORE_PRECISION,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = DefaultDict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        stopwords = load_stopwords()
        for movie in movies:
            id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.docmap[id] = movie
            self.__add_document(id, text, stopwords)

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term, set())
        return sorted(list(ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        if token not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        doc_ids = self.index[token]
        term_match_doc_count = len(doc_ids)
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

        return idf

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = (
            1 - b + b * (doc_length / avg_doc_length) if avg_doc_length != 0 else 1
        )

        bm25tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25tf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if not tokens:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_count = len(self.index[token])
        return math.log(
            (total_doc_count - term_match_count + 0.5) / (term_match_count + 0.5) + 1
        )

    def bm25(self, doc_id, term) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = tokenize(query)
        scores: dict[int, float] = defaultdict(lambda: 0)

        for token in tokens:
            doc_ids = self.index[token]
            for id in doc_ids:
                bm25 = self.bm25(id, token)
                scores[id] += bm25

        top_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        result = []
        for doc_id, score in top_items:
            item = dict()
            item["id"] = doc_id
            item["score"] = round(score, SCORE_PRECISION)
            item["title"] = self.docmap[doc_id]["title"]
            item["document"] = self.docmap[doc_id]["description"]
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def __add_document(self, doc_id, text, stopwords) -> None:
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total = sum(self.doc_lengths.values())

        return total / len(self.doc_lengths)


def search_command(query):
    inverted_index = InvertedIndex()
    inverted_index.load()
    stopwords = load_stopwords()
    results, seen = [], set()

    query_tokens = tokenize(query, stopwords)
    for token in query_tokens:
        docs = inverted_index.get_documents(token)
        for id in docs:
            if id in seen:
                continue
            seen.add(id)
            movie = inverted_index.docmap[id]
            results.append(movie)
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                return results
    return results


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def tf_command(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()

    frequency = inverted_index.get_tf(doc_id, term)
    return frequency


def idf_command(term) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_idf(term)


def tfidf_command(doc_id, term) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_tfidf(doc_id, term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b=BM25_B) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_bm25_tf(doc_id, term, k1, b)


def bm25_idf_command(term) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_bm25_idf(term)


def bm25_search_command(query: str, limit=5) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.bm25_search(query, limit)


def preprocess_text(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text.lower()


def tokenize(text: str, stopwords: list[str] = []) -> list[str]:
    stemmer = PorterStemmer()
    clean_text = preprocess_text(text)
    tokens = clean_text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens
