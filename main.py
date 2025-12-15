import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from src import store
from src.pdf_utils import extract_pdf_chunks
from src.image_utils import load_image_bytes, is_image_file


load_dotenv()


def parse_topics(topics_str: str | None) -> list[str]:
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
    print("Top results:")
    for item in results:
        meta = item.get("metadata", {})
        print(f"- path={meta.get('path')} score={item.get('score')} topics={meta.get('topics')}")


def _maybe_move_file(path: Path, topics: list[str], base_dir: Path):
    if not topics:
        return
    target_dir = base_dir / topics[0]
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
            chunks = extract_pdf_chunks(str(path))
            payloads = [
                {"text": c, "path": str(path), "topics": topics, "chunk_id": f"{path}:{i}"}
                for i, c in enumerate(chunks)
            ]
            store.add_document_chunks(payloads)
            moved = _maybe_move_file(path, topics, organized_dir)
            print(f"Ingested PDF {path} ({len(payloads)} chunks) -> {moved}")
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

