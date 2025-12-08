import string, pickle, os
from typing import DefaultDict
from nltk.stem import PorterStemmer
from lib.search_utils import CACHE_PATH, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = DefaultDict(set)
        self.docmap: dict[int, dict] = {}

    def build(self) -> None:
        movies = load_movies()
        stopwords = load_stopwords()
        for movie in movies:
            id = movie["id"]
            text = f"{movie["title"]} {movie["description"]}"
            self.docmap[id] = movie
            self.__add_document(id, text, stopwords)

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)

        index_path = os.path.join(CACHE_PATH, 'index.pkl')
        docmap_path = os.path.join(CACHE_PATH, 'docmap.pkl')

        with open(index_path, 'wb') as f1, open(docmap_path, 'wb') as f2:
            pickle.dump(self.index, f1)
            pickle.dump(self.docmap, f2)


    def get_documents(self, term: str) -> list[int]:
        ids = self.index.get(term, set())
        return sorted(list(ids))


    def __add_document(self, doc_id, text, stopwords):
        tokens = tokenize(text, stopwords)
        for token in tokens:
            self.index[token].add(doc_id)

def search_command(query):
    result = []
    movies = load_movies()
    stopwords = load_stopwords()

    for movie in movies:
        query_tokens = tokenize(query, stopwords)
        title_tokens = tokenize(movie["title"], stopwords)

        if has_matching(query_tokens, title_tokens):
            result.append(movie)
            if len(result) >= DEFAULT_SEARCH_LIMIT:
                break

    result.sort(key=lambda movie: movie["title"])
    return result

def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    docs = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def has_matching(query_tokens: list[str], title_tokens: list[str]):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

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

