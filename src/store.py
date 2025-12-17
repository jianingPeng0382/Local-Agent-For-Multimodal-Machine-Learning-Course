import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.embeddings import get_text_embedding, get_image_embedding, get_text_embeddings_batch


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
    # 确保使用与文本embedding相同的维度（1536 for text-embedding-3-small）
    # 如果collection已存在但维度不匹配，需要重新创建
    try:
        collection = client.get_collection(name=IMG_COLLECTION)
        # 检查维度是否匹配（1536是text-embedding-3-small的维度）
        # 如果collection为空或维度不匹配，删除并重新创建
        count = collection.count()
        if count > 0:
            # 获取一个样本检查维度
            sample = collection.get(limit=1)
            if sample['embeddings'] and len(sample['embeddings'][0]) != 1536:
                # 维度不匹配，删除旧collection
                client.delete_collection(name=IMG_COLLECTION)
                collection = client.create_collection(name=IMG_COLLECTION)
        return collection
    except Exception:
        # Collection不存在，创建新的
        return client.create_collection(name=IMG_COLLECTION)


def add_document_chunks(chunks: List[Dict[str, Any]]) -> None:
    """chunks: list of {text, path, topics, chunk_id}"""
    if not chunks:
        return
    collection = _doc_collection()
    
    # Prepare data
    valid_chunks = []
    texts = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        valid_chunks.append((idx, chunk))
        texts.append(text)
    
    if not texts:
        return
    
    # Get embeddings in batch (more efficient and reduces API calls)
    embeddings_batch = get_text_embeddings_batch(texts)
    
    # Process results
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for (idx, chunk), emb in zip(valid_chunks, embeddings_batch):
        if not emb:  # Skip failed embeddings
            continue
        doc_id = chunk.get("chunk_id") or f"{chunk.get('path','unknown')}:{idx}"
        ids.append(doc_id)
        documents.append(chunk.get("text", "").strip())
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


def add_image(path: str, image_bytes: bytes, topics: Optional[List[str]] = None) -> None:
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


def get_image_count() -> int:
    """返回图像集合中的图像数量"""
    collection = _img_collection()
    return collection.count()


def _expand_query_for_search(query: str) -> str:
    """扩展查询文本，使其包含更多相关关键词，提高搜索准确性"""
    query_lower = query.lower().strip()
    
    # 定义主题扩展映射
    expansions = {
        "nlp": "natural language processing text analysis language models linguistics transformers attention mechanism sequence models",
        "cv": "computer vision image recognition object detection visual processing convolutional neural networks vision models",
        "ml": "machine learning neural networks deep learning algorithms",
        "ai": "artificial intelligence machine learning neural networks",
    }
    
    # 检查查询是否匹配某个主题
    for key, expansion in expansions.items():
        if key in query_lower or query_lower in key:
            return f"{query} {expansion}"
    
    # 如果没有匹配，返回原查询
    return query


def query_images(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    collection = _img_collection()
    # 扩展查询以提高搜索准确性
    expanded_query = _expand_query_for_search(query)
    emb = get_text_embedding(expanded_query)
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

