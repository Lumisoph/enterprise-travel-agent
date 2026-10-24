"""RAG 工程边界回归：不依赖网络、向量模型或在线 LLM。"""
import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rag_retrieval import load_corpus, BM25Index, fuse_rankings, pack_evidence, validate_citations

spec = importlib.util.spec_from_file_location("rag_stub", ROOT / "tests/agentscope_stub.py")
stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stub)
stub.install()
spec = importlib.util.spec_from_file_location("rag_test_actual", ROOT / ".claude/skills/ask-question/script/agent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def document(identifier=1, content="北京住宿500元，超标需要审批"):
    return {"id": identifier, "content": content, "metadata": {"source": "样例.txt", "section": "住宿标准"}}


class RAGTests(unittest.TestCase):
    def test_sections_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "a.txt").write_text("测试制度\n\n一、住宿\n\n500元\n\n二、审批\n\n主管审批", encoding="utf-8")
            docs = load_corpus(folder)
            self.assertEqual(docs, load_corpus(folder))
            self.assertFalse(any("500元" in d["content"] and "主管审批" in d["content"] for d in docs))
            self.assertEqual(docs[-1]["metadata"]["section"], "二、审批")

    def test_budget_dedup_complete_blocks(self):
        doc = document()
        selected, context = pack_evidence([doc, document(2)], 200)
        self.assertEqual(len(selected), 1)
        self.assertIn(doc["content"], context)
        self.assertLessEqual(len(context), 200)
        self.assertEqual(pack_evidence([doc], 5), ([], ""))

    def test_rrf_uses_ranks_not_score_scale(self):
        a, b = document(1), document(2, "报销30天")
        result = fuse_rankings([{**a, "distance": .1}, b], [{**b, "bm25_score": 999999}], 2)
        self.assertEqual(result[0]["id"], 2)
        self.assertEqual(len(fuse_rankings([a, a], [], 3)), 1)

    def test_bm25_numbers_and_terms(self):
        hits = BM25Index([document(), document(2, "报销30天")]).search("住宿500元")
        self.assertEqual(hits[0]["id"], 1)

    def test_citations(self):
        docs, _ = pack_evidence([document()])
        self.assertTrue(validate_citations("限额500元[E1]", docs)["valid"])
        self.assertFalse(validate_citations("限额500元[E9]", docs)["valid"])
        self.assertFalse(validate_citations("限额500元", docs)["valid"])

    def agent(self, model=None):
        agent = module.RAGKnowledgeAgent.__new__(module.RAGKnowledgeAgent)
        agent.initialized = True
        agent.name = "测试"
        agent.model = model
        agent.context_max_chars = 2400
        agent._trace = threading.local()
        agent.skill_loader = type("Loader", (), {"get_skill_content": lambda *_: "基于证据回答"})()
        agent.search_knowledge = lambda _: [document()]
        return agent

    def call(self, agent):
        return json.loads(asyncio.run(agent.reply(stub.Msg("用户", "住宿标准", "user"))).content)

    def test_retrieval_failure_distinct_from_empty(self):
        agent = self.agent()
        def fail(_):
            raise RuntimeError("连接失败")
        agent.search_knowledge = fail
        self.assertEqual(self.call(agent)["error"], "retrieval_error")
        agent.search_knowledge = lambda _: []
        self.assertEqual(self.call(agent)["status"], "no_knowledge")

    def test_generation_failure(self):
        async def fail(_):
            raise RuntimeError("服务不可用")
        self.assertEqual(self.call(self.agent(fail))["error"], "generation_error")

    def test_generated_citation_validation(self):
        async def valid(_):
            return {"content": "北京住宿500元[E1]"}
        async def invalid(_):
            return {"content": "北京住宿500元[E99]"}
        self.assertEqual(self.call(self.agent(valid))["status"], "success")
        self.assertEqual(self.call(self.agent(invalid))["error"], "invalid_citations")

    def test_no_llm_returns_traceable_evidence(self):
        result = self.call(self.agent())
        self.assertEqual(result["retrieved_documents"][0]["citation_id"], "E1")
        self.assertTrue(result["citation_check"]["valid"])

    def test_empty_query_rejected_before_io(self):
        agent = self.agent()
        with self.assertRaises(ValueError):
            module.RAGKnowledgeAgent.search_knowledge(agent, "  ")


if __name__ == "__main__":
    unittest.main()
