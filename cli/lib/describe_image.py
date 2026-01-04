import mimetypes, os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def describe_image(image: str, query: str):
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image file not found: {image}")

    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"
    prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
    """

    with open(image, "rb") as file:
        img = file.read()

    parts = [
        prompt,
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(model=model, contents=parts)

    if response.text is None:
        raise RuntimeError("No text in Gemini response")

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
