import json, os

PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DOCUMENT_PREVIEW_LENGTH = 100

def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, 'r') as file:
        content = file.read()
        return content.splitlines()
