"""固定人工证据集上的真实本地检索对照，不调用生成模型。"""
import argparse
import importlib.util
from importlib.metadata import version
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rag_retrieval import load_corpus, BM25Index, fuse_rankings

# 每道题以必要证据字符串标注相关片段；不以整篇同源文档命中代替证据命中。
CASES = [
    ("北京出差住宿每晚限额多少？", ["500元"]),
    ("高原地区住宿标准如何调整？", ["20%"]),
    ("差旅申请需要提前多少个工作日？", ["3个工作日"]),
    ("出差结束后多久提交报销？", ["30个自然日"]),
    ("12月份出差最晚什么时候报销？", ["1月15日"]),
    ("纸质发票保存多少年？", ["7年"]),
    ("发票抬头写错且无法重开怎么办？", ["无法重开的"]),
    ("酒店minibar可以报销吗？", ["minibar"]),
    ("国际漫游业务通话报销需要什么？", ["通话记录"]),
    ("航班误机，国内国际分别提前多久到机场？", ["1.5小时", "3小时"]),
    ("航班延误报销要索取什么证明？", ["延误证明"]),
    ("托运行李没到，应获取什么编号？", ["报告编号"]),
    ("境外领事保护热线12308是什么？", ["12308"]),
    ("到店无房时酒店费用差额谁承担？", ["费用差额"]),
    ("额外购买的保险可以报销吗？", ["额外购买的保险"]),
    ("个人原因产生的改签费能报销吗？", ["个人原因产生的改签费"]),
]


def load_agent():
    # 无框架安装时只替代消息与基类；编码模型和 Milvus 均为真实依赖。
    try:
        import agentscope
    except ImportError:
        spec = importlib.util.spec_from_file_location("offline_stub", ROOT / "tests/agentscope_stub.py")
        stub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stub)
        stub.install()
    spec = importlib.util.spec_from_file_location("rag_actual", ROOT / ".claude/skills/ask-question/script/agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/rag_engineering")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    documents = load_corpus(ROOT / ".claude/skills/ask-question/data/documents")
    index = BM25Index(documents)
    Agent = load_agent()
    # 每次评测生成独立库，不修改历史工具回放或默认库。
    agent = Agent(knowledge_base_path=str(output / f"index_{time.time_ns()}"))
    try:
        result = agent.add_documents(documents)
        if result["status"] != "success":
            raise RuntimeError(result)
        # 真实持久化幂等验证，不依赖模拟客户端。
        repeated = agent.add_documents(documents)
        if repeated["status"] != "success" or repeated["total_count"] != len(documents):
            raise RuntimeError(f"重复写入验证失败：{repeated}")
        agent.retrieval_mode = "hybrid"
        if not agent.search_knowledge("北京住宿标准"):
            raise RuntimeError("实际混合检索链路未返回证据")
        agent.retrieval_mode = "dense"
        rows = []
        agent.search_knowledge("差旅住宿")  # 编码器与索引预热，不计入延迟。
        for query, evidence in CASES:
            gold = {d["id"] for d in documents if all(term in d["content"] for term in evidence)}
            if not gold:
                raise ValueError(f"人工证据未匹配到语料：{query}")
            start = time.perf_counter()
            dense = agent.search_knowledge(query, top_k=10)
            dense_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            lexical = index.search(query, 10)
            lexical_ms = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            hybrid = fuse_rankings(dense, lexical, 3)
            hybrid_ms = dense_ms + lexical_ms + (time.perf_counter()-start)*1000
            for mode, hits, latency in (("dense", dense[:3], dense_ms), ("bm25", lexical[:3], lexical_ms), ("hybrid", hybrid, hybrid_ms)):
                ranks = [i for i,d in enumerate(hits, 1) if d["id"] in gold]
                rows.append({"query": query, "evidence": evidence, "mode": mode,
                             "hit_at_3": int(bool(ranks)), "recall_at_3": len(ranks)/len(gold),
                             "rr_at_3": 1/min(ranks) if ranks else 0, "latency_ms": latency,
                             "gold_ids": sorted(gold), "retrieved_ids": [d["id"] for d in hits]})
        summary = {mode: {key: sum(r[key] for r in rows if r["mode"] == mode)/len(CASES)
                          for key in ("hit_at_3", "recall_at_3", "rr_at_3", "latency_ms")}
                   for mode in ("dense", "bm25", "hybrid")}
        report = {"documents": 8, "chunks": len(documents), "questions": len(CASES),
                  "index_path": str(agent.knowledge_base_path.relative_to(ROOT)),
                  "embedding": {"name": "bge-small-zh-v1.5", "pooling": "CLS", "normalize": True, "max_seq_length": 512},
                  "versions": {name: version(name) for name in ("torch", "sentence-transformers", "transformers", "pymilvus", "milvus-lite")},
                  "integration_checks": ["真实BGE CLS池化及归一化编码", "真实Milvus写入与检索", "重复upsert行数不增加", "实际hybrid检索入口"],
                  "scope": "本地小样本检索开发集，人工必要证据字符串弱标注；不是独立测试集，不包含生成评测；框架可能使用消息桩",
                  "latency_scope": "单次预热后；dense 为10候选召回再截Top3，hybrid为顺序双路总耗时；非服务P95",
                  "summary": summary, "details": rows}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"chunks": len(documents), "summary": summary}, ensure_ascii=False, indent=2))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
