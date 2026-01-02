import os, time

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def llm_rerank_individual(query: str, docs: list[dict], limit: int = 5):
    for doc in docs:
        prompt = f"""Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Give me ONLY the number in your response, no other text or explanation.

    Score:"""
        response = client.models.generate_content(model=model, contents=prompt)
        new_score = (response.text or "").strip().strip('"')
        doc["individual_score"] = float(new_score) if new_score else doc["rrf_score"]
        time.sleep(3)
    docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return docs[:limit]


def rerank(query: str, docs: list[dict], method: str = "batch", limit: int = 5):
    match method:
        case "individual":
            llm_rerank_individual(query, docs, limit)
        case _:
            pass
