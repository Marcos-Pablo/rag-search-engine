import json
import os

PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, 'r') as file:
        content = file.read()
        return content.splitlines()
