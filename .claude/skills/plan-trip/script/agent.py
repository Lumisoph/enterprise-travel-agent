"""
行程规划智能体
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict, Any
import json
import logging
import re
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# project_root = Path(__file__).parent.parent # Removed old logic
# sys.path.insert(0, str(project_root))

from utils.json_parser import robust_json_parse, extract_json_from_async_response

logger = logging.getLogger(__name__)

# 解析失败后的有限重试次数（总尝试次数 = 1 + 该值）
PARSE_RETRY_LIMIT = 1

# 重试时追加到提示词末尾的约束：强调只输出合法、完整闭合的 JSON
RETRY_HINT = ("\n\n【重要】上一次输出无法解析为合法 JSON（可能因输出过长被截断）。"
              "请只输出一个完整、闭合的 JSON 对象：不要 Markdown 代码块、不要任何解释文字；"
              "如内容较长，请精简描述，确保 JSON 在输出上限内完整结束，"
              "字符串内不要出现未转义的换行或尾随逗号。")


def parse_itinerary_json(text: str) -> dict:
    """把模型输出解析为行程 JSON：标准清洗 + 前缀 JSON 兜底。

    与改造前相同的两级解析策略，抽成函数供有限重试复用。
    """
    try:
        result = robust_json_parse(text, fallback=None)
    except Exception:
        clean_text = text
        if "```" in clean_text:
            clean_text = re.sub(r'```json\s*', '', clean_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'```', '', clean_text)
        clean_text = clean_text.strip()
        start_idx = clean_text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object start '{' found")
        obj, _ = json.JSONDecoder().raw_decode(clean_text[start_idx:])
        result = obj
    if result is None:
        raise ValueError("Parsed result is None")
    return result


class ItineraryPlanningAgent(AgentBase):
    """
    行程规划智能体（主协调）
    职责：协调事项收集、路线规划、酒店规划等多个子任务

    整合三层编排智能体的结果，生成完整行程计划
    """

    def __init__(self, name: str = "ItineraryPlanningAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content={}, role="assistant")

        # 解析输入内容
        content = x.content if not isinstance(x, list) else x[-1].content

        # 初始化变量
        user_query = ""
        context_info = {}
        previous_results = []
        user_preferences = {}

        # 如果content是JSON字符串，解析它（来自OrchestrationAgent）
        if isinstance(content, str):
            try:
                data = json.loads(content)
                context_info = data.get("context", {})
                user_query = context_info.get("rewritten_query", "")
                previous_results = data.get("previous_results", [])
                user_preferences = context_info.get("user_preferences", {})
            except json.JSONDecodeError:
                user_query = content
        elif isinstance(content, dict):
            context_info = content
            user_query = content.get("rewritten_query", str(content))
            user_preferences = content.get("user_preferences", {})

        # 整合所有可用信息
        all_info = {
            "user_query": user_query,
            "context": context_info,
        }

        # 从previous_results中提取其他agent的数据
        for prev in previous_results:
            agent_name = prev.get("agent_name", "")
            result_data = prev.get("result", {}).get("data", {})
            if result_data and agent_name:
                all_info[agent_name] = result_data

        # 构建用户偏好信息
        preferences_info = ""
        if user_preferences:
            pref_parts = ["【用户偏好】（规划时优先考虑）"]
            if user_preferences.get("home_location"):
                pref_parts.append(f"• 家庭住址: {user_preferences['home_location']}")
            if user_preferences.get("hotel_brands"):
                pref_parts.append(f"• 酒店偏好: {', '.join(user_preferences['hotel_brands'])}")
            if user_preferences.get("airlines"):
                pref_parts.append(f"• 航空偏好: {', '.join(user_preferences['airlines'])}")
            if user_preferences.get("seat_preference"):
                pref_parts.append(f"• 座位偏好: {user_preferences['seat_preference']}")

            if len(pref_parts) > 1:
                preferences_info = "\n".join(pref_parts) + "\n\n"

        # 获取当前时间
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        current_month = datetime.now().month
        current_season = "冬季" if current_month in [12, 1, 2] else \
                        "春季" if current_month in [3, 4, 5] else \
                        "夏季" if current_month in [6, 7, 8] else "秋季"
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 尝试从 SKILL.md 动态读取详细指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("plan-trip")
        if not skill_instruction:
            # Fallback: 如果读取失败，使用默认的简单指令
            skill_instruction = "请根据用户需求和偏好生成行程规划。"

        prompt = f"""你是一个高级行程规划专家。

【当前时间】
{current_date} {weekday}，当前季节是{current_season}

【用户需求】
{user_query}

{preferences_info}【所有收集的信息】
{json.dumps(all_info, ensure_ascii=False, indent=2)}

【任务说明与指南】
{skill_instruction}

请直接输出 JSON 格式的行程规划。
"""

        # 有限重试：解析失败后用补充了格式约束的提示再调用一次，仍失败才走兜底。
        # 重试位于产品公共执行路径，串行/并行两种调度策略共用同一规则；
        # 重试产生的模型调用耗时会自然计入节点耗时与批次耗时。
        parse_attempts = 0
        retried = False
        text = ""
        try:
            result = None
            parse_error = None
            for attempt in range(PARSE_RETRY_LIMIT + 1):
                parse_attempts = attempt + 1
                attempt_prompt = prompt if attempt == 0 else prompt + RETRY_HINT
                # 调用模型 - 使用消息列表格式（模型调用本身失败不重试，与原行为一致）
                response = await self.model([
                    {"role": "user", "content": attempt_prompt}
                ])

                # 获取响应文本
                text = await extract_json_from_async_response(response)

                try:
                    result = parse_itinerary_json(text)
                    break
                except Exception as exc:
                    parse_error = exc
                    logger.warning(
                        "行程 JSON 解析失败（第 %d/%d 次尝试）：%s；原始响应前 500 字符：%s",
                        parse_attempts, PARSE_RETRY_LIMIT + 1, exc, str(text)[:500])
                    if attempt < PARSE_RETRY_LIMIT:
                        retried = True

            if result is None:
                raise ValueError(
                    f"JSON 解析失败：共尝试 {parse_attempts} 次，"
                    f"最后一次错误：{parse_error}")

            # 记录重试次数（仅在实际重试时写入，正常路径输出保持不变）
            if retried and isinstance(result, dict):
                result["parse_retry_count"] = parse_attempts - 1

        except Exception as e:
            logger.error(f"Itinerary planning failed: {e}")
            logger.error(f"Raw response text (first 500 chars): {str(text)[:500]}")

            # 构建用户友好的错误消息
            error_detail = str(e)
            if "JSON" in error_detail or "parse" in error_detail.lower():
                user_message = "抱歉，模型返回的数据格式有误，无法解析行程信息。请稍后重试或简化您的需求描述。"
            else:
                user_message = f"行程规划过程中出现问题：{error_detail}"

            result = {
                "itinerary": {
                    "title": "行程规划",
                    "duration": "待完善",
                    "daily_plans": []
                },
                "planning_complete": False,
                "error": user_message,
                "technical_error": str(e),  # 保留技术细节用于调试
                # 便于事后判断格式错误还是输出截断：记录尝试次数与响应尾部
                "parse_attempts": parse_attempts,
                "raw_response_chars": len(text) if isinstance(text, str) else None,
                "raw_response_tail": str(text)[-200:],
            }

        # 返回JSON字符串格式
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")
