import string
from lib.search_utils import load_movies

DEFAULT_SEARCH_LIMIT = 5

def search_command(query):
    result = []
    movies = load_movies()

    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocessed_title = preprocess_text(movie["title"])

        if preprocessed_query in preprocessed_title:
            result.append(movie)

        if len(result) >= DEFAULT_SEARCH_LIMIT:
            break

    result.sort(key=lambda movie: movie["title"])
    return result

def preprocess_text(text: str):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table).lower()
