import json
import os

PROJECT_ROOT = os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    return data["movies"]

