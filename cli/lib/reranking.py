import os, time, json

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def llm_rerank_individual(query: str, docs: list[dict], limit: int = 5):
    scored_docs = []
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
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        time.sleep(3)
    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return docs[:limit]


def llm_rerank_batch(query: str, docs: list[dict], limit: int = 5):
    doc_list = []
    doc_map = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else, do not even wrap the answer with formatting like ```json```. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (response.text or "").strip()
    parsed_ids = json.loads(ranking_text)
    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append(
                {**doc_map[doc_id], "batch_score": i + 1},
            )

    return reranked[:limit]


def rerank(
    query: str, docs: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    match method:
        case "individual":
            return llm_rerank_individual(query, docs, limit)
        case "batch":
            return llm_rerank_batch(query, docs, limit)
        case _:
            return docs[:limit]
