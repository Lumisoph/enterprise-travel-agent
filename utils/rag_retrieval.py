"""差旅 RAG 的文档处理、词法召回、排名融合与证据装配。"""
import hashlib
import math
import re
from collections import Counter
from pathlib import Path


def tokenize(text):
    """中文双字滑窗保留局部词序，英文和数字按完整词匹配，无分词依赖。"""
    terms = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower()):
        terms.extend([part] if len(part) == 1 or not re.fullmatch(r"[\u4e00-\u9fff]+", part)
                     else [part[i:i + 2] for i in range(len(part) - 1)])
    return terms


def load_corpus(directory, max_chars=400):
    """按章节及段落切块，保留标题、文件、章节、块序号和内容哈希。"""
    if max_chars < 100:
        raise ValueError("切块上限不能小于100字符")
    documents = []
    for path in sorted(Path(directory).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = text.splitlines()[0].strip()
        section, paragraphs = title, []
        parts = []

        def flush():
            chunk = ""
            for paragraph in paragraphs:
                # 超长段落按行优先切分；最后才使用定长分割。
                units = [paragraph] if len(paragraph) <= max_chars else paragraph.splitlines()
                for unit in units:
                    for start in range(0, len(unit), max_chars):
                        piece = unit[start:start + max_chars]
                        if chunk and len(chunk) + len(piece) + 2 > max_chars:
                            parts.append((section, chunk))
                            chunk = ""
                        chunk = (chunk + "\n\n" + piece).strip()
            if chunk:
                parts.append((section, chunk))

        for paragraph in re.split(r"\n\s*\n", text):
            if re.match(r"^[一二三四五六七八九十百]+、", paragraph.strip()):
                flush()
                paragraphs = []
                section = paragraph.splitlines()[0].strip()
            paragraphs.append(paragraph.strip())
        flush()
        for index, (heading, content) in enumerate(parts):
            digest = hashlib.sha256((path.name + "\0" + content).encode()).hexdigest()
            documents.append({
                "id": int(digest[:15], 16),
                "content": content,
                "metadata": {"title": title, "source": path.name, "parent_doc": path.name,
                             "section": heading, "chunk_index": index,
                             "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                             "source_kind": "本地示例文档，非已核验的企业现行政策"},
            })
    return documents


class BM25Index:
    """面向小型本地语料的 BM25，初始化统计一次，查询只遍历内存。"""
    def __init__(self, documents):
        self.documents = documents
        self.counts = [Counter(tokenize(d["metadata"].get("title", "") + " "
                                        + d["metadata"].get("section", "") + " " + d["content"]))
                       for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(documents))
        self.df = Counter(term for counts in self.counts for term in counts)

    def search(self, query, limit=10):
        terms = set(tokenize(query))
        hits = []
        for document, counts, length in zip(self.documents, self.counts, self.lengths):
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if frequency:
                    idf = math.log(1 + (len(self.documents) - self.df[term] + 0.5)
                                   / (self.df[term] + 0.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (
                        0.25 + 0.75 * length / (self.average or 1)))
            if score > 0:
                hits.append({**document, "bm25_score": score})
        return sorted(hits, key=lambda d: (-d["bm25_score"], d["id"]))[:limit]


def fuse_rankings(dense, lexical, limit=3, rank_constant=60):
    """RRF 只融合名次，不直接相加量纲不同的余弦与 BM25 分数。"""
    merged = {}
    for channel, hits in (("dense", dense), ("lexical", lexical)):
        seen = set()
        for rank, document in enumerate(hits, 1):
            key = document["id"]
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, {**document, "rrf_score": 0.0, "ranks": {}})
            item["rrf_score"] += 1 / (rank_constant + rank)
            item["ranks"][channel] = rank
            if "distance" in document:
                item["distance"] = document["distance"]
    return sorted(merged.values(), key=lambda d: (-d["rrf_score"], d["id"]))[:limit]


def pack_evidence(documents, max_chars=2400):
    """按完整片段装配证据，预算含来源标签；不截断金额与适用条件。"""
    selected, blocks, used, seen = [], [], 0, set()
    for document in documents:
        fingerprint = hashlib.sha256(document["content"].encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        metadata = document.get("metadata", {})
        citation = f"E{len(selected) + 1}"
        header = f"[{citation}] 来源：{metadata.get('source', metadata.get('parent_doc', '未标注'))}；章节：{metadata.get('section', metadata.get('title', '未标注'))}"
        block = header + "\n" + document["content"]
        cost = len(block) + (2 if blocks else 0)
        if used + cost > max_chars:
            continue
        selected.append({**document, "citation_id": citation})
        blocks.append(block)
        used += cost
    return selected, "\n\n".join(blocks)


def validate_citations(answer, documents):
    """检查引用存在性；不将引用存在误当作语义蕴含正确。"""
    cited = set(re.findall(r"\[(E\d+)\]", answer))
    available = {d["citation_id"] for d in documents}
    return {"cited": sorted(cited), "invalid": sorted(cited - available),
            "valid": bool(cited) and not (cited - available)}
