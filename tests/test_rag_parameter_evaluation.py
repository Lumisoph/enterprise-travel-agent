"""验证参数实验口径，避免分母变化及留出集泄漏。"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_rag_parameters import evidence_metrics, choose, configurations
from scripts.rag_parameter_cases import DEV, TEST
from utils.rag_retrieval import pack_evidence, validate_citations


class EvaluationTests(unittest.TestCase):
    def test_split_quote_counts_as_missing(self):
        case = {"evidence": [{"source": "a", "quote": "审批后报销"}]}
        documents = [{"id": 1, "content": "审批后", "metadata": {"source": "a"}},
                     {"id": 2, "content": "报销", "metadata": {"source": "a"}}]
        self.assertEqual(evidence_metrics(case, documents)["all_evidence"], 0)

    def test_source_is_part_of_gold(self):
        case = {"evidence": [{"source": "a", "quote": "500元"}]}
        self.assertEqual(evidence_metrics(case, [{"content": "500元", "metadata": {"source": "b"}}])["any_evidence"], 0)

    def test_selector_ignores_test(self):
        settings = configurations()[:2]
        results = [{"valid": True, "dev": {"all_evidence": 1, "evidence_coverage": 1, "context_chars": 300}, "test": {"all_evidence": 0}},
                   {"valid": True, "dev": {"all_evidence": .5, "evidence_coverage": .5, "context_chars": 200}, "test": {"all_evidence": 1}}]
        before = choose(results, settings)
        changed = copy.deepcopy(results)
        changed[0]["test"]["all_evidence"] = 1
        changed[1]["test"]["all_evidence"] = 0
        self.assertEqual(before, choose(changed, settings))

    def test_holdout_quotes_not_identical_to_dev(self):
        dev = {(a["source"], a["quote"]) for c in DEV for a in c["evidence"]}
        test = {(a["source"], a["quote"]) for c in TEST for a in c["evidence"]}
        self.assertFalse(dev & test)

    def test_complete_pack_vs_naive_truncation(self):
        document = {"id": 1, "content": "可报销" + "说明" * 100 + "必须主管审批", "metadata": {"source": "样例"}}
        _, context = pack_evidence([document], 100)
        self.assertEqual(context, "")
        self.assertIn("可报销", document["content"][:100])
        self.assertNotIn("必须主管审批", document["content"][:100])

    def test_existing_citation_not_semantic_guarantee(self):
        selected, _ = pack_evidence([{"id": 1, "content": "住宿500元", "metadata": {}}])
        self.assertTrue(validate_citations("住宿800元[E1]", selected)["valid"])


if __name__ == "__main__":
    unittest.main()
