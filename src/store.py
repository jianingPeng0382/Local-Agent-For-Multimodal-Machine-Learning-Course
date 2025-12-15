import os
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from src.embeddings import get_text_embedding, get_image_embedding


INDEX_PATH = os.path.join("data", "index")
DOC_COLLECTION = "documents"
IMG_COLLECTION = "images"


def _client() -> chromadb.PersistentClient:
    os.makedirs(INDEX_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=INDEX_PATH)


def _doc_collection():
    client = _client()
    return client.get_or_create_collection(name=DOC_COLLECTION)


def _img_collection():
    client = _client()
    return client.get_or_create_collection(name=IMG_COLLECTION)


def add_document_chunks(chunks: List[Dict[str, Any]]) -> None:
    """chunks: list of {text, path, topics, chunk_id}"""
    if not chunks:
        return
    collection = _doc_collection()
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        emb = get_text_embedding(text)
        if not emb:
            continue
        doc_id = chunk.get("chunk_id") or f"{chunk.get('path','unknown')}:{idx}"
        ids.append(doc_id)
        documents.append(text)
        metadatas.append(
            {
                "path": chunk.get("path", ""),
                "topics": ",".join(chunk.get("topics", [])),
            }
        )
        embeddings.append(emb)
    if not ids:
        return
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    collection = _doc_collection()
    emb = get_text_embedding(query)
    if not emb:
        return []
    res = collection.query(query_embeddings=[emb], n_results=top_k)
    results = []
    for i in range(len(res.get("ids", [])[0])):
        results.append(
            {
                "id": res["ids"][0][i],
                "score": res["distances"][0][i] if "distances" in res else None,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
            }
        )
    return results


def add_image(path: str, image_bytes: bytes, topics: List[str] | None = None) -> None:
    if not image_bytes:
        return
    collection = _img_collection()
    emb = get_image_embedding(image_bytes)
    if not emb:
        return
    topics = topics or []
    collection.add(
        ids=[path],
        embeddings=[emb],
        metadatas=[{"path": path, "topics": ",".join(topics)}],
    )


def query_images(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    collection = _img_collection()
    emb = get_text_embedding(query)
    if not emb:
        return []
    res = collection.query(query_embeddings=[emb], n_results=top_k)
    results = []
    for i in range(len(res.get("ids", [])[0])):
        results.append(
            {
                "id": res["ids"][0][i],
                "score": res["distances"][0][i] if "distances" in res else None,
                "metadata": res["metadatas"][0][i],
            }
        )
    return results

