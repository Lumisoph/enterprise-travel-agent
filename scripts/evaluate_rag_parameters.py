"""冻结协议后，用真实 BGE/Milvus 补充参数敏感性和设计消融。"""
import hashlib
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_rag_engineering import load_agent
from scripts.rag_parameter_cases import DEV, TEST, NO_ANSWER
from utils.rag_retrieval import load_corpus, BM25Index, fuse_rankings, pack_evidence, validate_citations

OUTPUT = ROOT / "experiments/rag_parameters"
CORPUS = ROOT / ".claude/skills/ask-question/data/documents"


@dataclass(frozen=True)
class Setting:
    chunk: int = 400
    splitter: str = "chapter"
    pooling: str = "cls"
    normalize: bool = True
    mode: str = "hybrid"
    candidate: int = 10
    top_k: int = 3
    rrf: int = 60
    budget: int = 2400
    titles: bool = True


def configurations():
    base = Setting()
    settings = [base]
    for key, values in {"chunk": [200, 300, 500, 600], "candidate": [3, 5, 20],
                        "top_k": [1, 5, 8], "rrf": [10, 30, 100],
                        "budget": [600, 1200, 3600], "mode": ["dense", "bm25"],
                        "splitter": ["fixed", "overlap"], "pooling": ["mean"],
                        "normalize": [False], "titles": [False]}.items():
        settings.extend(replace(base, **{key: value}) for value in values)
    return settings


def corpus_for(setting):
    if setting.splitter == "chapter":
        return load_corpus(CORPUS, setting.chunk)
    documents = []
    step = setting.chunk - (80 if setting.splitter == "overlap" else 0)
    for path in sorted(CORPUS.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        for i, start in enumerate(range(0, len(text), step)):
            content = text[start:start+setting.chunk]
            identifier = int(hashlib.sha256((path.name+"\0"+content).encode()).hexdigest()[:15], 16)
            documents.append({"id": identifier, "content": content,
                              "metadata": {"source": path.name, "title": text.splitlines()[0],
                                           "section": "固定字符窗口", "chunk_index": i}})
    return documents


def evidence_metrics(case, documents):
    """固定原文证据为分母；被切碎且不在同一片段中时算未覆盖，不丢弃问题。"""
    def matches(atom, doc):
        return atom["source"] == doc["metadata"]["source"] and atom["quote"] in doc["content"]
    covered = [any(matches(atom, doc) for doc in documents) for atom in case["evidence"]]
    first = next((i for i, doc in enumerate(documents, 1)
                  if any(matches(atom, doc) for atom in case["evidence"])), None)
    return {"evidence_coverage": sum(covered)/len(covered), "all_evidence": int(all(covered)),
            "any_evidence": int(any(covered)), "mrr": 1/first if first else 0}


def summarize(rows):
    keys = ("evidence_coverage", "all_evidence", "any_evidence", "mrr", "context_chars", "selected_chunks")
    return {key: statistics.mean(r[key] for r in rows) for key in keys}


def choose(results, settings):
    # 仅使用开发集：先完整证据率，再证据覆盖，再更少上下文；剩余按候选开销、登记顺序决胜。
    eligible = [i for i, result in enumerate(results) if result.get("valid")]
    return max(eligible, key=lambda i: (results[i]["dev"]["all_evidence"],
                                      results[i]["dev"]["evidence_coverage"],
                                      -results[i]["dev"]["context_chars"],
                                      -settings[i].candidate, -i))


def bootstrap_delta(baseline, selected, key="all_evidence"):
    deltas = [b[key]-a[key] for a,b in zip(baseline, selected)]
    generator = random.Random(20261023)
    means = sorted(statistics.mean(generator.choices(deltas, k=len(deltas))) for _ in range(2000))
    return {"mean": statistics.mean(deltas), "paired_bootstrap_95_ci": [means[49], means[1949]], "resamples": 2000}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = configurations()
    cases = DEV + TEST
    for case in cases:
        for atom in case["evidence"]:
            if atom["quote"] not in (CORPUS/atom["source"]).read_text(encoding="utf-8"):
                raise ValueError(f"原文证据不存在：{case['id']} {atom}")
    protocol = {"settings": [asdict(s) for s in settings], "dev": DEV, "test": TEST,
                "no_answer": NO_ANSWER,
                "selection": "仅dev：完整证据率最大，其次必要证据覆盖，再少上下文、少候选、先登记配置",
                "scope": "单因素敏感性，不是完整网格或全局最优；24题新留出由同一助手按新条款编写，非人类独立盲评",
                "corpus_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CORPUS.glob('*.txt'))}}
    # 在模型运行及观测结果前落盘。协议与源码共同提供可审计定义。
    protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2)
    (OUTPUT/"protocol.json").write_text(protocol_text, encoding="utf-8")
    protocol_hash = hashlib.sha256(protocol_text.encode()).hexdigest()
    Agent = load_agent()
    agent = Agent(knowledge_base_path=str(OUTPUT/f"index_{time.time_ns()}"))
    from sentence_transformers import SentenceTransformer, models
    queries = [c["query"] for c in cases] + NO_ANSWER
    model_by_pool = {"cls": agent.embedding_model}
    transformer = agent.embedding_model[0]
    model_by_pool["mean"] = SentenceTransformer(modules=[transformer, models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")])
    query_cache, cache = {}, {}
    results, details = [], []
    try:
        for i, setting in enumerate(settings):
            key = (setting.chunk, setting.splitter, setting.pooling, setting.normalize)
            if key not in cache:
                documents = corpus_for(setting)
                model = model_by_pool[setting.pooling]
                token_lengths = [len(ids) for ids in model.tokenizer([d["content"] for d in documents], truncation=False, padding=False)["input_ids"]]
                overlong = sum(n > model.max_seq_length for n in token_lengths)
                if overlong:
                    cache[key] = {"valid": False, "chunks": len(documents), "overlong": overlong, "max_tokens": max(token_lengths)}
                else:
                    collection = f"variant_{len(cache)}"
                    agent.milvus_client.create_collection(collection_name=collection, dimension=agent.embedding_dim, metric_type="COSINE", auto_id=False)
                    start = time.perf_counter()
                    vectors = model.encode([d["content"] for d in documents], normalize_embeddings=setting.normalize)
                    build_ms = (time.perf_counter()-start)*1000
                    agent.milvus_client.upsert(collection_name=collection, data=[{"id": d["id"], "vector": v.tolist()} for d,v in zip(documents,vectors)])
                    qkey = (setting.pooling, setting.normalize)
                    if qkey not in query_cache:
                        start = time.perf_counter()
                        query_cache[qkey] = model.encode(queries, normalize_embeddings=setting.normalize)
                        encoding_ms = (time.perf_counter()-start)*1000/len(queries)
                    else:
                        encoding_ms = None
                    mapping = {d["id"]: d for d in documents}
                    dense = []
                    latencies = []
                    agent.milvus_client.search(collection_name=collection, data=[query_cache[qkey][0].tolist()], limit=20)
                    # 查询向量离线缓存；仅比较真实库查询耗时，不冒充在线端到端延迟。
                    for repeat in range(3):
                        for qi, vector in enumerate(query_cache[qkey]):
                            start = time.perf_counter()
                            hits = agent.milvus_client.search(collection_name=collection, data=[vector.tolist()], limit=20)[0]
                            latencies.append((time.perf_counter()-start)*1000)
                            if repeat == 0:
                                dense.append([{**mapping[h["id"]], "distance": h["distance"]} for h in hits])
                    cache[key] = {"valid": True, "documents": documents, "dense": dense,
                                  "chunks": len(documents), "max_tokens": max(token_lengths),
                                  "build_encode_ms": build_ms, "query_batch_encode_ms_per_query": encoding_ms,
                                  "milvus_ms_median": statistics.median(latencies), "milvus_ms_p95": sorted(latencies)[int(.95*(len(latencies)-1))]}
            cached = cache[key]
            result = {"configuration_id": i, "setting": asdict(setting),
                      **{k:v for k,v in cached.items() if k not in ("documents", "dense")}}
            if not cached["valid"]:
                result["reason"] = "超出模型token上限，按产品入库规则拒绝；不以静默截断结果参加选参"
                results.append(result)
                print(f"配置{i}无效：超长片段{cached['overlong']}", flush=True)
                continue
            lexical_docs = cached["documents"] if setting.titles else [{**d, "metadata": {**d["metadata"], "title": "", "section": ""}} for d in cached["documents"]]
            index = BM25Index(lexical_docs)
            rows = []
            ranking_start = time.perf_counter()
            for qi, case in enumerate(cases):
                dense = cached["dense"][qi][:setting.candidate]
                lexical = index.search(case["query"], setting.candidate)
                # 去标题消融只影响词法打分，证据装配保持相同元数据。
                original = {d["id"]: d for d in cached["documents"]}
                lexical = [{**d, "metadata": original[d["id"]]["metadata"]} for d in lexical]
                hits = (dense[:setting.top_k] if setting.mode == "dense" else lexical[:setting.top_k]
                        if setting.mode == "bm25" else fuse_rankings(dense, lexical, setting.top_k, setting.rrf))
                selected, context = pack_evidence(hits, setting.budget)
                row = {"configuration_id": i, "case_id": case["id"], "split": case["split"],
                       **evidence_metrics(case, selected), "context_chars": len(context),
                       "selected_chunks": len(selected), "retrieval_ids": [d["id"] for d in hits],
                       "selected_ids": [d["id"] for d in selected],
                       "before_budget": evidence_metrics(case, hits)}
                rows.append(row)
            result["lexical_fusion_pack_ms_per_query"] = (time.perf_counter()-ranking_start)*1000/len(cases)
            result["dev"] = summarize(rows[:len(DEV)])
            result["test"] = summarize(rows[len(DEV):])
            details.extend(rows)
            results.append(result)
            print(f"配置{i}完成 dev完整证据率={result['dev']['all_evidence']:.3f}", flush=True)
        winner = choose(results, settings)
        baseline_rows = [r for r in details if r["configuration_id"] == 0 and r["split"] == "test"]
        winner_rows = [r for r in details if r["configuration_id"] == winner and r["split"] == "test"]
        no_answer = [{"query": q, "returned": len(cache[(400,"chapter","cls",True)]["dense"][len(cases)+i]),
                      "max_cosine": cache[(400,"chapter","cls",True)]["dense"][len(cases)+i][0]["distance"]}
                     for i,q in enumerate(NO_ANSWER)]
        report = {"protocol_sha256": protocol_hash, "selected_by_dev": winner,
                  "test_paired_delta_vs_hybrid_baseline": bootstrap_delta(baseline_rows, winner_rows),
                  "no_answer_probe": no_answer,
                  "results": results, "details": details,
                  "limitations": ["新留出非外部人类盲评，条款和来源分布与dev不同", "单因素比较不估计交互，不证明全局最优",
                                  "必要原文字符串是可审计代理指标，不是答案语义或最终合规", "缓存查询向量，库耗时不是在线端到端延迟",
                                  "未调用生成模型，未验证提示词、事实蕴含或拒答效果"]}
        (OUTPUT/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"selected": winner, "setting": asdict(settings[winner]), "dev": results[winner]["dev"],
                          "test": results[winner]["test"], "delta": report["test_paired_delta_vs_hybrid_baseline"]}, ensure_ascii=False, indent=2))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
