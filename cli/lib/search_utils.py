import json, os

PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5

def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, 'r') as file:
        content = file.read()
        return content.splitlines()
