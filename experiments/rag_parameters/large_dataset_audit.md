# 大规模 RAG 测试集审计

- 题目数：1000
- 证据组数：84
- 分层切分（按证据组）：{'train': 600, 'test': 228, 'dev': 172}

## 分布

- 类型：{'paraphrase': 1000}
- 来源：{'01_travel_standards.txt': 1000}

## 质量门禁

- 通过：all_evidence_present_in_source
- 通过：no_empty_query
- 未通过：has_multiple_intent_types
- 未通过：has_negative_cases
- 通过：split_by_evidence_group

## 主要问题

- 规范化重复问题：0 组
- 重复证据片段：84 组
- 证据/空题/来源错误：0 条

## 解释

- 同一 evidence 组的改写题必须放在同一 split；否则指标会被相似改写高估。
- 仅有 paraphrase 类型或没有负样本时，不应宣称测试了拒答能力。
- 证据存在性检查是必要条件，不能替代人工核验答案蕴含关系。
