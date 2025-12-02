import string
from lib.search_utils import load_movies

DEFAULT_SEARCH_LIMIT = 5

def search_command(query):
    result = []
    movies = load_movies()

    for movie in movies:
        query_tokens = tokenize(query)
        title_tokens = tokenize(movie["title"])

        if has_matching(query_tokens, title_tokens):
            result.append(movie)
            if len(result) >= DEFAULT_SEARCH_LIMIT:
                break

    result.sort(key=lambda movie: movie["title"])
    return result

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

def tokenize(text: str):
    clean_text = preprocess_text(text)
    tokens = clean_text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens
