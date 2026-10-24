"""按章节构建独立 RAG 索引；不覆盖或删除旧知识库。"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from utils.rag_retrieval import load_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="尚不存在的输出目录；验证后在调用方切换路径")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("输出目录已存在，请选择新目录以避免混合旧索引")
    documents = load_corpus(Path(__file__).resolve().parents[1] / "data/documents")
    if not documents:
        parser.error("未读取到有效文档")
    spec = importlib.util.spec_from_file_location("rag_build", Path(__file__).with_name("agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.RAGKnowledgeAgent(knowledge_base_path=str(output))
    try:
        result = agent.add_documents(documents)
        if result["status"] != "success":
            raise RuntimeError(result.get("message", "索引写入失败"))
        print(f"已构建 {len(documents)} 个章节片段：{output}")
        for document in agent.search_knowledge("出差住宿标准", top_k=3):
            print(f"{document['metadata'].get('source')} 余弦相似度：{document['distance']:.3f}")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
