from typing import List, Tuple, Optional
import re

from pypdf import PdfReader


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def extract_title_and_abstract(path: str) -> Tuple[Optional[str], Optional[str]]:
    """提取PDF的标题和摘要（通常在第一页）
    
    Returns:
        (title, abstract): 标题和摘要的元组
    """
    try:
        reader = PdfReader(path)
        if len(reader.pages) == 0:
            return None, None
        
        # 获取第一页文本
        first_page_text = reader.pages[0].extract_text() or ""
        if not first_page_text.strip():
            return None, None
        
        lines = first_page_text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # 改进的标题提取：跳过机构名、作者名等
        title = None
        for i, line in enumerate(lines[:20]):  # 检查前20行
            line_clean = line.strip()
            # 标题特征：长度适中，不包含特殊字符，不是明显的机构/作者格式
            if (15 <= len(line_clean) <= 150 and 
                '@' not in line_clean and 
                'university' not in line_clean.lower() and
                'department' not in line_clean.lower() and
                not line_clean.lower().startswith(('abstract', 'introduction', 'keywords', '1.', '2.')) and
                not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', line_clean) and  # 跳过"John Smith"格式
                not re.match(r'^\d+$', line_clean)):  # 跳过纯数字
                title = line_clean
                break
        
        # 提取摘要
        abstract = None
        abstract_started = False
        abstract_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # 查找"Abstract"标记
            if 'abstract' in line_lower and len(line) < 50:
                abstract_started = True
                continue
            
            if abstract_started:
                # 摘要通常在Abstract标记后，直到Introduction或Keywords
                if any(keyword in line_lower for keyword in ['introduction', 'keywords', '1.', '1 introduction', '2.']):
                    break
                if line and len(line) > 15:  # 跳过太短的行
                    abstract_lines.append(line)
                    if len(abstract_lines) >= 15:  # 摘要可能较长
                        break
        
        if abstract_lines:
            abstract = ' '.join(abstract_lines[:15])[:1500]  # 增加长度限制
        
        return title, abstract
    
    except Exception:
        return None, None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(chunk_size - overlap, 1)
    return chunks


def extract_pdf_chunks(path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = load_pdf_text(path)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)

