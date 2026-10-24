"""由实际实验结果生成中文报告，不改写历史实验数字。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "experiments/rag_parameters"
report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
base = protocol["settings"][0]
lines = ["# RAG 参数敏感性与设计消融实验", "", "日期：2026-10-23。以下数字来自本地真实 BGE＋Milvus，本轮不调用在线生成模型。", "",
         "## 原先验证到哪一步", "", "此前只有固定参数下的16题三路检索比较，以及10项工程回归；400字符、候选10、Top-3、RRF常数60、预算2400均未经过参数对照。此次补充24个单因素配置，覆盖这些参数以及切块策略、池化方式、归一化、词法标题元数据。", "",
         "## 协议与可比性", "",
         "- 原16题为开发集；新增24题作为本轮留出集，不用于选参。新题由同一助手依据其他条款编写，非外部独立盲评，也不代表真实用户分布。",
         "- 每题固定来源文件和必要原文片段，以必要证据为分母；所有切块方式使用同一标注。片段被切断且不能完整定位时记为未覆盖，不删除问题。",
         "- 完整证据率：装入上下文的片段覆盖该题全部必要原文的题目比例；证据覆盖率：每题必要原文覆盖比例宏平均。MRR按首个含至少一个必要原文的片段计算。不是生成准确率。",
         "- 先落盘protocol.json，再编码和执行。开发集选参规则：完整证据率优先，其次证据覆盖率，再少上下文、少候选、先登记配置。无全网格搜索，不估计参数交互。",
         "- 每个不同切块/池化/归一化配置真实建库；查询向量缓存，最多20候选后截取以比较候选数。库延迟是预热后3次重复、单独Milvus查询，不包含编码或生成；不能用该延迟证明候选3比10更快。",
         "", f"协议SHA256：`{report['protocol_sha256']}`", "", "## 完整配置结果", "",
         "参照配置：章节400字符、CLS、归一化、hybrid、每路10候选、Top-3、RRF=60、预算2400、词法层带标题章节。表中只列相对参照的单项变化。", "",
         "| 编号 | 单项变化 | 块数 | dev完整证据率 | 留出完整证据率 | 留出MRR | 留出平均上下文字符 |", "|---|---|---:|---:|---:|---:|---:|"]
for result in report["results"]:
    setting = result["setting"]
    diff = "; ".join(f"{k}={v}" for k,v in setting.items() if v != base[k]) or "参照配置"
    if not result["valid"]:
        lines.append(f"| {result['configuration_id']} | {diff} | {result['chunks']} | 无效 | 超token上限 | — | — |")
        continue
    dev, test = result["dev"], result["test"]
    lines.append(f"| {result['configuration_id']} | {diff} | {result['chunks']} | {dev['all_evidence']:.2%} | {test['all_evidence']:.2%} | {test['mrr']:.4f} | {test['context_chars']:.1f} |")
winner = report["selected_by_dev"]
delta = report["test_paired_delta_vs_hybrid_baseline"]
lines += ["", "## 解释与决策", "",
          f"1. 开发集规则选中配置{winner}（300字符），dev完整证据率100%，上下文804.4字符；留出23/24，低于400字符参照的24/24。配对差异为{delta['mean']:.2%}，2000次配对bootstrap区间为[{delta['paired_bootstrap_95_ci'][0]:.2%}, {delta['paired_bootstrap_95_ci'][1]:.2%}]。不宣称300最优，不改默认切块。该区间只反映这24题的重采样变动，不能解决题目来源偏差。",
          "2. Top-1留出完整证据率19/24；Top-3、5、8均24/24。更大K增加上下文，未体现必要证据收益，支持当前Top-3的局部取舍。",
          "3. 候选3、5、10留出均24/24，20为23/24；更大候选并非单调更好。不能证明10最优，暂留10为工程默认。",
          "4. RRF常数10、30、60、100在本题集结果相同。60属于当前可用默认，实验未证明它比其他常数好。",
          "5. 600字符预算留出21/24，1200、2400、3600均24/24；该设置下1200已足够，2400为余量，不是质量最优或token最优。",
          "6. 章节切块留出24/24，固定字符窗口22/24，80字符重叠窗口23/24；CLS为24/24，mean为23/24。仅在当前数据上支持保留章节边界和正确CLS池化。",
          "7. COSINE下归一化开关的排序与证据指标相同，符合余弦消除向量模长的性质，不应包装成归一化带来的准确率提升。",
          "8. hybrid留出24/24，dense为20/24，BM25为23/24；但BM25的MRR更高，指标之间存在取舍。新题较多词面明确的政策条款，不能推出混合检索普遍优于纯向量。",
          "9. 去掉词法标题章节后完整证据率不变，MRR从0.8819降至0.8333；这仅支持排序上的局部作用。",
          "10. 不更改业务默认：本轮检索参数仅有局部证据，缺真实框架/生成/最终行程评价。400、Top-3等保留；dense默认也不因这一批小型人工题直接自动切换。", "",
          "## 无答案探测与工程规则", "",
          "4个无答案/时效越界探测都返回了20个近邻（如截Top-3仍有结果），最高余弦在0.480至0.582之间。说明非空检索库的最近邻不能作为有答案判据。本轮未用这4题拟合阈值，未宣称已实现语义拒答。", "",
          "新增6项评测口径与设计测试，覆盖来源约束、切碎证据计为未覆盖、选择器忽略留出结果、开发与留出原文证据不重叠、完整片段装配对比机械截断、引用存在不保证语义正确。与原10项工程回归一起共16项。", "",
          "证据预算、错误分层、引用编号检查是可验证的工程规则；目前不能把它们说成经过真实LLM对照证明的幻觉降低。引用检查可让‘800元[E1]’通过，即使E1写的是500元，测试明确记录了这一边界。", "",
          "## 复现与产物", "", "```powershell", "& '.venv-rag\\Scripts\\python.exe' scripts\\evaluate_rag_parameters.py", "python scripts\\summarize_rag_parameters.py", "python -m unittest discover -s tests -p 'test_rag_*engineering.py' -v", "python -m unittest discover -s tests -p test_rag_parameter_evaluation.py -v", "```", "",
          "原始协议、语料哈希、题目来源证据、逐题配置结果分别见同目录protocol.json和report.json。原先16题检索报告仍保留，不替换其指标定义。"]
(folder / "实验报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("已生成中文实验报告")
