import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, List
import numpy as np

from dotenv import load_dotenv

from src import store
from src.embeddings import get_text_embedding
from src.pdf_utils import extract_pdf_chunks, extract_title_and_abstract
from src.image_utils import load_image_bytes, is_image_file


load_dotenv()


def parse_topics(topics_str: Optional[str]) -> List[str]:
    if not topics_str:
        return []
    return [t.strip() for t in topics_str.split(",") if t.strip()]


def cmd_add_paper(args):
    pdf_path = Path(args.path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found")
    topics = parse_topics(args.topics)
    chunks = extract_pdf_chunks(str(pdf_path))
    chunk_payloads = [
        {
            "text": chunk,
            "path": str(pdf_path),
            "topics": topics,
            "chunk_id": f"{pdf_path}:{i}",
        }
        for i, chunk in enumerate(chunks)
    ]
    store.add_document_chunks(chunk_payloads)
    print(f"Ingested {len(chunk_payloads)} chunks from {pdf_path}")


def cmd_search_paper(args):
    results = store.query_documents(args.query, top_k=args.top_k)
    print("Top results:")
    for item in results:
        meta = item.get("metadata", {})
        text = item.get("text") or ""
        snippet = text[:120].replace("\n", " ")
        print(
            f"- path={meta.get('path')} score={item.get('score')} topics={meta.get('topics')} "
            f"snippet={snippet}"
        )


def cmd_search_image(args):
    results = store.query_images(args.query, top_k=args.top_k)
    
    if not results:
        print("No images found.")
        return
    
    print(f"Found {len(results)} images for query: '{args.query}'")
    print("Top results:")
    
    # 根据搜索查询推断相关的topics
    inferred_topics = _infer_topics_from_query(args.query)
    
    # 创建结果文件夹（基于搜索查询）
    query_safe = "".join(c for c in args.query if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
    query_safe = query_safe.replace(' ', '_')
    results_dir = Path("search_results") / query_safe
    results_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    for i, item in enumerate(results, 1):
        meta = item.get("metadata", {})
        original_path = meta.get('path')
        score = item.get('score')
        
        # 显示结果（使用推断的topics而不是存储的topics）
        print(f"{i}. path={original_path} score={score:.4f} inferred_topics={','.join(inferred_topics) if inferred_topics else 'N/A'}")
        
        # 复制文件到结果文件夹
        if original_path and Path(original_path).exists():
            try:
                src_path = Path(original_path)
                # 保持原文件名，但添加序号前缀
                dst_path = results_dir / f"{i:02d}_{src_path.name}"
                shutil.copy2(str(src_path), str(dst_path))
                copied_files.append(str(dst_path))
            except Exception as e:
                print(f"  Warning: Could not copy {original_path}: {e}")
    
    if copied_files:
        print(f"\n✓ Copied {len(copied_files)} images to: {results_dir}")
    else:
        print("\n⚠ No images were copied (files may not exist)")


def _infer_topics_from_query(query: str) -> List[str]:
    """根据搜索查询推断相关的topics"""
    query_lower = query.lower()
    
    # CV相关关键词（更全面的列表）
    cv_keywords = [
        "visual", "vision", "image", "images", "picture", "pictures",
        "photo", "photograph", "chart", "graph", "diagram", "plot",
        "visualization", "visualize", "drawing", "illustration",
        "sunset", "sunrise", "landscape", "portrait", "scene", "scenery",
        "object", "detection", "recognition", "camera", "view", "viewpoint"
    ]
    
    # NLP相关关键词
    nlp_keywords = [
        "text", "language", "word", "sentence", "document", "paper",
        "translation", "transformer", "attention", "sequence",
        "linguistic", "semantic", "syntax", "grammar"
    ]
    
    inferred = []
    cv_score = sum(1 for keyword in cv_keywords if keyword in query_lower)
    nlp_score = sum(1 for keyword in nlp_keywords if keyword in query_lower)
    
    if cv_score > 0:
        inferred.append("CV")
    if nlp_score > 0:
        inferred.append("NLP")
    
    # 如果没有匹配，返回空列表（显示为N/A）
    return inferred


def _expand_topic_description(topic: str) -> str:
    """将简短的topic扩展为更详细的描述，提高分类准确性"""
    topic_lower = topic.lower().strip()
    expansions = {
        "cv": "computer vision image recognition object detection visual processing convolutional neural networks vision models",
        "nlp": "natural language processing text analysis language models linguistics transformers attention mechanism sequence models",
        "ml": "machine learning neural networks deep learning algorithms",
        "ai": "artificial intelligence machine learning neural networks",
    }
    
    # 如果topic是缩写，使用扩展描述
    if topic_lower in expansions:
        return f"{topic} {expansions[topic_lower]}"
    
    # 否则直接返回原topic
    return topic


def _classify_by_keywords(text: str, topics: List[str]) -> Optional[str]:
    """基于关键词的备选分类方法"""
    text_lower = text.lower()
    
    # CV相关关键词（包括图像生成、视觉处理等）
    cv_keywords = [
        "visual", "vision", "image", "images", "picture", "pictures",
        "convolutional", "cnn", "object detection", "recognition",
        "pixel", "pixels", "camera", "photograph", "photography",
        "visual instruction", "visual language", "multimodal vision",
        "generative adversarial", "gan", "adversarial", "generative model",
        "image generation", "image synthesis", "discriminative model"
    ]
    
    # NLP相关关键词
    nlp_keywords = [
        "language", "text", "translation", "transformer", "attention",
        "sequence", "transduction", "encoder", "decoder", "bert", "gpt",
        "natural language", "linguistic", "sentence", "word", "token",
        "attention is all you need", "sequence to sequence", "seq2seq"
    ]
    
    topic_keywords = {
        "CV": cv_keywords,
        "NLP": nlp_keywords,
    }
    
    # 计算每个topic的关键词匹配数
    topic_scores = {}
    for topic in topics:
        keywords = topic_keywords.get(topic.upper(), [])
        score = sum(1 for keyword in keywords if keyword in text_lower)
        topic_scores[topic] = score
    
    # 返回得分最高的topic（如果有明显差异）
    if topic_scores:
        max_score = max(topic_scores.values())
        if max_score > 0:
            # 如果最高分明显高于其他（至少多1分），使用关键词分类
            sorted_scores = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_scores) > 1:
                if sorted_scores[0][1] > sorted_scores[1][1]:  # 只要有差异就使用
                    return sorted_scores[0][0]
            elif len(sorted_scores) == 1:
                return sorted_scores[0][0]
    
    return None


def _classify_document_by_content(text_chunks: List[str], topics: List[str], title: Optional[str] = None, abstract: Optional[str] = None, debug: bool = False) -> str:
    """根据文档内容与topics的相似度，返回最匹配的topic
    
    使用多种策略：
    1. 首先尝试基于关键词的分类（快速且准确）
    2. 如果关键词分类不明确，使用embedding相似度
    
    Args:
        text_chunks: 文档的文本chunks
        topics: 可选的topics列表
        title: 文档标题（可选）
        abstract: 文档摘要（可选）
        debug: 是否输出调试信息
    """
    if not topics:
        return "Other"
    
    # 如果只有一个topic，直接返回
    if len(topics) == 1:
        return topics[0]
    
    # 构建用于分类的文档文本
    doc_parts = []
    if title:
        doc_parts.append(title)
    if abstract:
        doc_parts.append(abstract)
    if text_chunks:
        chunks_text = " ".join(text_chunks[:3])
        doc_parts.append(chunks_text)
    
    doc_text = " ".join(doc_parts)
    if len(doc_text) > 2000:
        if title and abstract:
            doc_text = f"{title} {abstract} {text_chunks[0] if text_chunks else ''}"[:2000]
        else:
            doc_text = doc_text[:2000]
    
    if not doc_text or not doc_text.strip():
        return topics[0]
    
    # 首先尝试基于关键词的分类
    keyword_result = _classify_by_keywords(doc_text, topics)
    if keyword_result:
        if debug:
            print(f"  Classified by keywords: {keyword_result}")
        return keyword_result
    
    # 如果关键词分类不明确，使用embedding相似度
    doc_embedding = get_text_embedding(doc_text)
    if not doc_embedding:
        return topics[0]
    
    best_topic = topics[0]
    best_similarity = -1
    similarities = {}
    
    # 计算文档与每个topic的相似度
    for topic in topics:
        expanded_topic = _expand_topic_description(topic)
        topic_embedding = get_text_embedding(expanded_topic)
        if not topic_embedding:
            continue
        
        doc_vec = np.array(doc_embedding)
        topic_vec = np.array(topic_embedding)
        
        similarity = np.dot(doc_vec, topic_vec) / (
            np.linalg.norm(doc_vec) * np.linalg.norm(topic_vec)
        )
        
        similarities[topic] = similarity
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_topic = topic
    
    if debug:
        print(f"  Classification by embedding - scores: {similarities}")
        print(f"  Selected: {best_topic} (similarity: {best_similarity:.4f})")
    
    return best_topic


def _maybe_move_file(path: Path, topics: list[str], base_dir: Path, text_chunks: List[str] = None, title: Optional[str] = None, abstract: Optional[str] = None, debug: bool = False):
    """根据内容相似度智能分类文件
    
    Args:
        path: 文件路径
        topics: topics列表
        base_dir: 基础目录
        text_chunks: 文档文本chunks（可选）
        title: 文档标题（可选）
        abstract: 文档摘要（可选）
        debug: 是否输出调试信息
    """
    if not topics:
        return path
    
    # 如果有多个topics且有文档内容，使用内容相似度分类
    if len(topics) > 1 and (text_chunks or title or abstract):
        target_topic = _classify_document_by_content(text_chunks or [], topics, title=title, abstract=abstract, debug=debug)
    else:
        target_topic = topics[0]
    
    target_dir = base_dir / target_topic
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / path.name
    try:
        shutil.move(str(path), str(target_path))
        return target_path
    except Exception:
        return path


def cmd_organize_folder(args):
    root = Path(args.dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")
    topics = parse_topics(args.topics)
    organized_dir = root / "organized"

    for path in root.iterdir():
        if path.is_dir():
            continue
        if path.suffix.lower() == ".pdf":
            # 提取标题和摘要（用于分类）
            title, abstract = extract_title_and_abstract(str(path))
            
            chunks = extract_pdf_chunks(str(path))
            payloads = [
                {"text": c, "path": str(path), "topics": topics, "chunk_id": f"{path}:{i}"}
                for i, c in enumerate(chunks)
            ]
            store.add_document_chunks(payloads)
            
            # 传递标题、摘要和chunks用于智能分类
            moved = _maybe_move_file(path, topics, organized_dir, text_chunks=chunks, title=title, abstract=abstract, debug=False)
            info = f"({len(payloads)} chunks"
            if title:
                info += f", title: {title[:50]}..."
            info += ")"
            print(f"Ingested PDF {path} {info} -> {moved}")
        elif is_image_file(str(path)):
            img_bytes = load_image_bytes(str(path))
            store.add_image(str(path), img_bytes, topics=topics)
            moved = _maybe_move_file(path, topics, organized_dir)
            print(f"Ingested image {path} -> {moved}")
    print("Organize complete.")


def cmd_init_git(_args):
    if not (Path(".git").exists()):
        os.system("git init")
        print("Initialized git repository.")
    else:
        print("Git repository already initialized.")
    print("Next steps: git add . && git commit -m \"init\"")


def build_parser():
    parser = argparse.ArgumentParser(description="Local multimodal assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    add_paper = sub.add_parser("add_paper", help="Ingest and tag a PDF")
    add_paper.add_argument("path", help="Path to PDF")
    add_paper.add_argument("--topics", default="", help="Comma-separated topics")
    add_paper.set_defaults(func=cmd_add_paper)

    search_paper = sub.add_parser("search_paper", help="Semantic search PDFs")
    search_paper.add_argument("query")
    search_paper.add_argument("--top_k", type=int, default=5)
    search_paper.set_defaults(func=cmd_search_paper)

    search_image = sub.add_parser("search_image", help="Text to image search")
    search_image.add_argument("query")
    search_image.add_argument("--top_k", type=int, default=5)
    search_image.set_defaults(func=cmd_search_image)

    organize = sub.add_parser("organize_folder", help="Batch ingest PDFs/images in folder")
    organize.add_argument("dir", help="Folder to scan")
    organize.add_argument("--topics", default="", help="Comma-separated topics to tag/move")
    organize.set_defaults(func=cmd_organize_folder)

    init_git = sub.add_parser("init_git", help="Initialize git repository")
    init_git.set_defaults(func=cmd_init_git)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

