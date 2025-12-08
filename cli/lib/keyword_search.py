import string, pickle, os, sys
from typing import DefaultDict
from nltk.stem import PorterStemmer
from lib.search_utils import CACHE_PATH, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = DefaultDict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_PATH, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_PATH, 'docmap.pkl')

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

        with open(self.index_path, 'wb') as f1, open(self.docmap_path, 'wb') as f2:
            pickle.dump(self.index, f1)
            pickle.dump(self.docmap, f2)

    def load(self) -> None:
        with open(self.index_path, 'rb') as f1, open(self.docmap_path, 'rb') as f2:
            self.index = pickle.load(f1)
            self.docmap = pickle.load(f2)

    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term, set())
        return sorted(list(ids))


    def __add_document(self, doc_id, text, stopwords) -> None:
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index[token].add(doc_id)

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

def preprocess_text(text: str):
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text.lower()

def tokenize(text: str, stopwords: list[str]):
    stemmer = PorterStemmer()
    clean_text = preprocess_text(text)
    tokens = clean_text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens

