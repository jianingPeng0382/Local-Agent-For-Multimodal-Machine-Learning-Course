import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

TEXT_EMBED_MODEL = "text-embedding-3-small"
IMAGE_EMBED_MODEL = "gpt-image-embedding-1"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Export it or add to .env.")
    return OpenAI(api_key=api_key)


def get_text_embedding(text: str) -> List[float]:
    client = _client()
    text = text.strip()
    if not text:
        return []
    resp = client.embeddings.create(input=text, model=TEXT_EMBED_MODEL)
    return resp.data[0].embedding


def get_image_embedding(image_bytes: bytes) -> List[float]:
    client = _client()
    resp = client.embeddings.create(
        input=image_bytes,
        model=IMAGE_EMBED_MODEL,
    )
    return resp.data[0].embedding

