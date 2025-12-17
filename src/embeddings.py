import os
import base64
import time
from functools import lru_cache
from typing import List, Optional
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError


load_dotenv()

TEXT_EMBED_MODEL = "text-embedding-3-small"
VISION_MODEL = "gpt-4o-mini"  # 使用支持视觉的模型
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 初始延迟（秒）
REQUEST_DELAY = 0.1  # 请求之间的延迟（秒）


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Export it or add to .env.")
    return OpenAI(api_key=api_key, timeout=60.0)


def get_text_embedding(text: str, retry_count: int = 0) -> List[float]:
    """Get text embedding using OpenAI API with retry mechanism"""
    client = _client()
    text = text.strip()
    if not text:
        return []
    
    try:
        # Add delay between requests to avoid rate limiting
        if retry_count == 0:
            time.sleep(REQUEST_DELAY)
        
        resp = client.embeddings.create(input=text, model=TEXT_EMBED_MODEL)
        return resp.data[0].embedding
    except (APIConnectionError, APITimeoutError, RateLimitError) as e:
        if retry_count < MAX_RETRIES:
            # Exponential backoff
            delay = RETRY_DELAY * (2 ** retry_count)
            print(f"API error (attempt {retry_count + 1}/{MAX_RETRIES}): {type(e).__name__}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            return get_text_embedding(text, retry_count + 1)
        else:
            print(f"Error generating text embedding after {MAX_RETRIES} retries: {e}")
            return []
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return []


def get_text_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call (batch processing)"""
    client = _client()
    
    # Filter empty texts
    valid_texts = [t.strip() for t in texts if t.strip()]
    if not valid_texts:
        return []
    
    try:
        # Add delay before batch request
        time.sleep(REQUEST_DELAY)
        
        # OpenAI embeddings API supports batch input
        resp = client.embeddings.create(input=valid_texts, model=TEXT_EMBED_MODEL)
        
        # Return embeddings in the same order as input
        embeddings = [item.embedding for item in resp.data]
        
        # Pad with empty lists for empty texts
        result = []
        text_idx = 0
        for text in texts:
            if text.strip():
                result.append(embeddings[text_idx])
                text_idx += 1
            else:
                result.append([])
        
        return result
    except (APIConnectionError, APITimeoutError, RateLimitError) as e:
        print(f"Batch API error: {type(e).__name__}. Falling back to individual requests...")
        # Fallback to individual requests
        return [get_text_embedding(text) for text in texts]
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return [get_text_embedding(text) for text in texts]


def get_image_embedding(image_bytes: bytes, retry_count: int = 0) -> List[float]:
    """Get image embedding using OpenAI API with retry mechanism
    
    Strategy: Use Vision API to get image description, then embed the description.
    This provides a semantic embedding that captures the image content.
    """
    client = _client()
    
    if not image_bytes:
        return []
    
    try:
        # Add delay between requests
        if retry_count == 0:
            time.sleep(REQUEST_DELAY)
        
        # Convert image bytes to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use Vision API to get detailed image description
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail, including all visible objects, text, colors, layout, and any other relevant information. If this appears to be from an academic paper, research, or scientific content, please identify the research field (e.g., NLP, computer vision, machine learning), any visible text, charts, graphs, tables, experimental results, or technical diagrams. If it contains text, transcribe key words and phrases. Provide a comprehensive description that would be useful for semantic search, especially for academic and research-related queries."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            timeout=60.0
        )
        
        # Get the image description
        image_description = response.choices[0].message.content.strip()
        
        # Embed the description using text embedding API
        embedding = get_text_embedding(image_description)
        
        return embedding
        
    except (APIConnectionError, APITimeoutError, RateLimitError) as e:
        if retry_count < MAX_RETRIES:
            # Exponential backoff
            delay = RETRY_DELAY * (2 ** retry_count)
            print(f"Image API error (attempt {retry_count + 1}/{MAX_RETRIES}): {type(e).__name__}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            return get_image_embedding(image_bytes, retry_count + 1)
        else:
            print(f"Error generating image embedding after {MAX_RETRIES} retries: {e}")
            return []
    except Exception as e:
        print(f"Error generating image embedding via API: {e}")
        return []

