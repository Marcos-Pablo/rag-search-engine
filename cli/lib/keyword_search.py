import string, pickle, os, math
from typing import Counter, DefaultDict
from nltk.stem import PorterStemmer
from nltk.util import defaultdict
from lib.search_utils import CACHE_PATH, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = DefaultDict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)

        self.index_path = os.path.join(CACHE_PATH, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_PATH, 'docmap.pkl')
        self.term_frequencies_path = os.path.join(CACHE_PATH, 'term_frequencies.pkl')


    def build(self) -> None:
        movies = load_movies()
        stopwords = load_stopwords()
        for movie in movies:
            id = movie["id"]
            text = f"{movie["title"]} {movie["description"]}"
            self.docmap[id] = movie
            self.__add_document(id, text, stopwords)

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)

        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)


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

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if not tokens:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_count = len(self.index[token])
        return math.log((total_doc_count - term_match_count + 0.5) / (term_match_count + 0.5) + 1)

    def __add_document(self, doc_id, text, stopwords) -> None:
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

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

def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def tf_command(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()

    frequency = inverted_index.get_tf(doc_id, term)
    return frequency

def idf_command(term):
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_idf(term)

def tfidf_command(doc_id, term):
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_tfidf(doc_id, term)

def bm25_idf_command(term):
    inverted_index = InvertedIndex()
    inverted_index.load()

    return inverted_index.get_bm25_idf(term)

def preprocess_text(text: str):
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

