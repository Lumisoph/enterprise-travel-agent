"""
协调器智能体 OrchestrationAgent
职责：根据意图识别结果，协调调度多个子智能体完成任务

核心功能：
1. 接收 IntentionAgent 的调度决策
2. 按照优先级顺序执行子智能体
3. 管理智能体之间的消息传递
4. 聚合多个智能体的结果
5. 与三层记忆系统集成

执行模式（调度策略开关 strategy）：
- sequential (顺序执行): 优先级批次之间顺序执行，每批内部也逐个顺序执行
- priority_parallel (同优先级并行，默认): 优先级批次之间顺序执行，每批内部并行执行

两种策略共用同一份 _prepare_context / _execute_agent / _aggregate_results / 记忆更新
与错误处理代码，只改变执行顺序与重叠程度。同一批的所有节点看到相同的初始上下文
和相同的前序批次结果快照：串行模式下，本批先产生的节点结果也不会传给本批其他节点。
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict, Any
import json
import logging
import asyncio
import time

# 支持的调度策略
STRATEGIES = ("sequential", "priority_parallel")

logger = logging.getLogger(__name__)


class OrchestrationAgent(AgentBase):
    """协调器智能体 - 调度和协调多个子智能体"""

    def __init__(
        self,
        name: str = "OrchestrationAgent",
        agent_registry: Dict[str, AgentBase] = None,
        memory_manager = None,
        strategy: str = "priority_parallel",
        **kwargs
    ):
        """
        初始化协调器

        Args:
            name: 智能体名称
            agent_registry: 子智能体注册表 {agent_name: agent_instance}
            memory_manager: 记忆管理器
            strategy: 调度策略，sequential 或 priority_parallel
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"未知调度策略：{strategy}，可选：{'、'.join(STRATEGIES)}")
        super().__init__()
        self.name = name
        self.agent_registry = agent_registry or {}
        self.memory_manager = memory_manager
        self.strategy = strategy
        # 执行轨迹：每次 reply 重置，记录每个节点的起止时间偏移与批次，
        # 用于证明并行重叠与批次等待（业务结果不依赖该轨迹）。
        self.execution_trace = []
        self._reply_started = 0.0

    def register_agent(self, agent_name: str, agent: AgentBase):
        """注册子智能体"""
        self.agent_registry[agent_name] = agent
        logger.info(f"Registered agent: {agent_name}")

    def unregister_agent(self, agent_name: str):
        """注销子智能体"""
        if agent_name in self.agent_registry:
            del self.agent_registry[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        协调执行流程

        Args:
            x: 输入消息，应包含 IntentionAgent 的输出

        Returns:
            Msg: 执行结果
        """
        if x is None:
            return Msg(
                name=self.name,
                content=json.dumps({"error": "No input provided"}),
                role="assistant"
            )

        # 每次执行重置轨迹
        self.execution_trace = []
        self._reply_started = time.perf_counter()

        # 解析输入
        if isinstance(x, list):
            intention_output = x[-1].content if x else "{}"
        else:
            intention_output = x.content

        # 解析意图识别结果
        try:
            intention_data = json.loads(intention_output) if isinstance(intention_output, str) else intention_output
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intention output: {e}")
            return Msg(
                name=self.name,
                content=json.dumps({"error": "Invalid intention format"}),
                role="assistant"
            )

        # 获取智能体调度计划
        agent_schedule = intention_data.get("agent_schedule", [])
        if not agent_schedule:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "status": "no_agents",
                    "message": "没有需要调度的智能体"
                }),
                role="assistant"
            )

        # 按优先级排序并分组为批次
        sorted_schedule = sorted(agent_schedule, key=lambda x: x.get("priority", 999))
        batches = self._group_by_priority(sorted_schedule)

        logger.info(f"Orchestrating {len(sorted_schedule)} agents "
                    f"in {len(batches)} priority batches (strategy={self.strategy})")

        # 准备上下文信息（两种策略共用）
        context = self._prepare_context(intention_data)

        # 逐批执行。每批共享同一份前序批次结果快照：
        # 批内节点（无论串行还是并行）都看不到本批其他节点的结果。
        results = []
        for batch_index, batch in enumerate(batches):
            snapshot = list(results)
            batch_results = await self._execute_batch(batch, batch_index, context, snapshot)
            results.extend(batch_results)

        # 聚合结果
        final_result = self._aggregate_results(results, intention_data)

        # 更新记忆
        if self.memory_manager:
            self._update_memory(intention_data, results)

        return Msg(
            name=self.name,
            content=json.dumps(final_result, ensure_ascii=False),
            role="assistant"
        )

    def _prepare_context(self, intention_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备上下文信息，供子智能体使用

        Args:
            intention_data: 意图识别结果

        Returns:
            上下文字典
        """
        context = {
            "reasoning": intention_data.get("reasoning", ""),
            "intents": intention_data.get("intents", []),
            "key_entities": intention_data.get("key_entities", {}),
            "rewritten_query": intention_data.get("rewritten_query", "")
        }

        # 从记忆系统获取上下文
        if self.memory_manager:
            # 短期记忆：最近对话
            recent_context = self.memory_manager.short_term.get_recent_context(3)
            context["recent_dialogue"] = recent_context

            # 长期记忆：用户偏好
            preferences = self.memory_manager.long_term.get_preference()
            context["user_preferences"] = preferences

        return context

    @staticmethod
    def _group_by_priority(sorted_schedule: List[Dict]) -> List[List[Dict]]:
        """
        把按优先级排序后的调度计划分组为批次

        Args:
            sorted_schedule: 已按优先级升序排序的任务列表

        Returns:
            批次列表，每批内的任务优先级相同
        """
        batches = []
        for task in sorted_schedule:
            if not batches or batches[-1][0].get("priority", 0) != task.get("priority", 0):
                batches.append([])
            batches[-1].append(task)
        return batches

    async def _execute_batch(
        self,
        tasks: List[Dict],
        batch_index: int,
        context: Dict[str, Any],
        previous_results: List[Dict]
    ) -> List[Dict]:
        """
        执行一个优先级批次

        两种策略共用同一任务封装（含错误兜底与轨迹记录），
        仅执行顺序与重叠程度不同：
        - priority_parallel: 批内任务并行执行
        - sequential: 批内任务逐个顺序执行

        Args:
            tasks: 本批任务列表
            batch_index: 批次序号（从 0 开始）
            context: 上下文信息（本批所有节点共用）
            previous_results: 前序批次结果快照（本批所有节点共用同一份）

        Returns:
            执行结果列表，结构与策略无关
        """
        if not tasks:
            return []

        async def run_task(task: Dict) -> Dict:
            agent_name = task.get("agent_name")
            priority = task.get("priority", 0)
            began = time.perf_counter()
            entry = {
                "agent_name": agent_name,
                "priority": priority,
                "batch_index": batch_index,
                "strategy": self.strategy,
                "start_offset": began - self._reply_started,
                # 是否真正进入智能体执行（区分「懒加载失败、未执行」与
                # 「执行了但结果失败」）。失败请求的耗时同样计入轨迹。
                "entered_execution": False,
            }
            try:
                # 单节点执行统一走 _execute_agent（其内部已兜底大部分异常）；
                # 此处再兜底注册表懒加载等异常，保证两种策略的观察行为一致。
                result = await self._execute_agent(
                    agent_name=agent_name,
                    context=context,
                    reason=task.get("reason", ""),
                    expected_output=task.get("expected_output", ""),
                    previous_results=previous_results,
                    trace_entry=entry
                )
            except Exception as exc:
                logger.error(f"Agent execution failed: {agent_name}, error: {exc}")
                result = {
                    "status": "error",
                    "agent_name": agent_name,
                    "data": {"error": str(exc)},
                    "message": f"智能体执行失败: {str(exc)}"
                }
            ended = time.perf_counter()
            entry.update({
                "end_offset": ended - self._reply_started,
                "duration": ended - began,
                "status": result.get("status", "unknown")
            })
            self.execution_trace.append(entry)
            return {"agent_name": agent_name, "priority": priority, "result": result}

        if self.strategy == "priority_parallel" and len(tasks) > 1:
            logger.info(f"Executing {len(tasks)} agents in parallel (batch={batch_index})")
            gathered = await asyncio.gather(*(run_task(task) for task in tasks), return_exceptions=True)
            results = []
            for task, item in zip(tasks, gathered):
                if isinstance(item, Exception):
                    logger.error(f"Parallel agent execution failed: {task.get('agent_name')}, error: {item}")
                    results.append({
                        "agent_name": task.get("agent_name"),
                        "priority": task.get("priority", 0),
                        "result": {
                            "status": "error",
                            "agent_name": task.get("agent_name"),
                            "data": {"error": str(item)},
                            "message": f"并行执行失败: {str(item)}"
                        }
                    })
                else:
                    results.append(item)
            return results

        # 单任务或 sequential：逐个顺序执行，但每个任务仍只看到
        # 前序批次的快照，看不到本批其他节点的结果。
        results = []
        for task in tasks:
            results.append(await run_task(task))
        return results

    async def _execute_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
        reason: str,
        expected_output: str,
        previous_results: List[Dict],
        trace_entry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行单个智能体

        Args:
            agent_name: 智能体名称
            context: 上下文信息
            reason: 调用原因
            expected_output: 期望输出
            previous_results: 前序智能体的结果

        Returns:
            执行结果
        """
        # 检查智能体是否注册
        if agent_name not in self.agent_registry:
            logger.warning(f"Agent not registered: {agent_name}")
            return {
                "status": "error",
                "message": f"智能体未注册: {agent_name}"
            }

        agent = self.agent_registry[agent_name]

        # 懒加载成功、即将真正执行：标记「已进入执行」，供实验区分
        # 未执行的加载失败与执行后失败（二者耗时与解读不同）。
        if trace_entry is not None:
            trace_entry["entered_execution"] = True

        # 构建输入消息
        input_msg = Msg(
            name="Orchestrator",
            content=json.dumps({
                "context": context,
                "reason": reason,
                "expected_output": expected_output,
                "previous_results": previous_results
            }, ensure_ascii=False),
            role="user"
        )

        try:
            # 调用智能体
            response = await agent.reply(input_msg)

            # 解析响应
            if isinstance(response.content, str):
                try:
                    result = json.loads(response.content)
                except json.JSONDecodeError:
                    result = {"output": response.content}
            else:
                result = response.content

            # 检查 result 中是否有 error 字段
            # 如果有，说明智能体内部执行失败了
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "未知错误")
                return {
                    "status": "error",
                    "agent_name": agent_name,
                    "data": result,
                    "message": error_msg
                }

            return {
                "status": "success",
                "agent_name": agent_name,
                "data": result
            }

        except Exception as e:
            logger.error(f"Agent execution failed: {agent_name}, error: {e}")
            # 返回友好的错误信息，但不中断流程
            return {
                "status": "error",
                "agent_name": agent_name,
                "data": {"error": str(e)},
                "message": f"智能体执行失败: {str(e)}"
            }

    def _aggregate_results(
        self,
        results: List[Dict],
        intention_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        聚合多个智能体的结果

        Args:
            results: 所有智能体的执行结果
            intention_data: 原始意图识别结果

        Returns:
            聚合后的最终结果
        """
        aggregated = {
            "status": "completed",
            "intention": {
                "intents": intention_data.get("intents", []),
                "key_entities": intention_data.get("key_entities", {})
            },
            "agents_executed": len(results),
            "results": []
        }

        # 收集每个智能体的结果
        for result in results:
            aggregated["results"].append({
                "agent_name": result["agent_name"],
                "priority": result["priority"],
                "status": result["result"].get("status", "unknown"),
                "data": result["result"].get("data", {})
            })

        # 检查是否有错误
        errors = [r for r in results if r["result"].get("status") == "error"]
        if errors:
            aggregated["status"] = "partial_failure"
            aggregated["errors"] = len(errors)

        return aggregated

    def _update_memory(self, intention_data: Dict[str, Any], results: List[Dict]):
        """
        更新记忆系统

        Args:
            intention_data: 意图识别结果
            results: 智能体执行结果
        """
        if not self.memory_manager:
            return

        # 提取并保存信息到长期记忆
        for result in results:
            agent_name = result["agent_name"]
            data = result["result"].get("data", {})

            # 如果是偏好智能体，保存偏好信息到长期记忆
            if agent_name == "preference" and isinstance(data, dict):
                preferences_data = data.get("preferences", {})

                # 新格式：preferences 是列表，包含 {type, value, action}
                if isinstance(preferences_data, list):
                    for pref_item in preferences_data:
                        if not isinstance(pref_item, dict):
                            continue

                        pref_type = pref_item.get("type")
                        pref_value = pref_item.get("value")
                        pref_action = pref_item.get("action", "replace")  # 默认覆盖

                        if not pref_type or not pref_value:
                            continue

                        # 根据 action 决定操作
                        if pref_action == "append":
                            # 追加模式：获取现有值并追加
                            current_prefs = self.memory_manager.long_term.get_preference()
                            existing_value = current_prefs.get(pref_type)

                            # 如果现有值是列表，追加
                            if isinstance(existing_value, list):
                                if pref_value not in existing_value:
                                    existing_value.append(pref_value)
                                self.memory_manager.long_term.save_preference(pref_type, existing_value)
                                logger.info(f"Appended to {pref_type}: {pref_value}, total: {existing_value}")
                            else:
                                # 如果现有值不是列表，创建新列表
                                new_list = [existing_value, pref_value] if existing_value else [pref_value]
                                self.memory_manager.long_term.save_preference(pref_type, new_list)
                                logger.info(f"Created list for {pref_type}: {new_list}")
                        else:
                            # 覆盖模式：直接保存新值
                            self.memory_manager.long_term.save_preference(pref_type, pref_value)
                            logger.info(f"Replaced {pref_type}: {pref_value}")

                # 旧格式兼容：preferences 是字典
                elif isinstance(preferences_data, dict):
                    for pref_type, value in preferences_data.items():
                        if value and pref_type != "has_preferences" and pref_type != "error":
                            self.memory_manager.long_term.save_preference(pref_type, value)
                            logger.info(f"Updated {pref_type}: {value} (legacy format)")

            # 如果是行程规划智能体，保存行程到长期记忆
            if agent_name == "itinerary_planning" and isinstance(data, dict):
                itinerary = data.get("itinerary", {})

                # 只要有行程信息就保存（不管是否完全规划好）
                if itinerary:
                    # 提取事项收集的信息（出发地、目的地等）
                    event_data = {}
                    for r in results:
                        if r["agent_name"] == "event_collection":
                            event_data = r["result"].get("data", {})
                            break

                    # 从 event_data 获取行程信息
                    origin = event_data.get("origin")
                    destination = event_data.get("destination")
                    start_date = event_data.get("start_date")
                    end_date = event_data.get("end_date")
                    purpose = event_data.get("trip_purpose", "旅游")

                    # 保存到长期记忆（只要有目的地就保存）
                    if destination:
                        self.memory_manager.long_term.save_trip_history({
                            "origin": origin,
                            "destination": destination,
                            "start_date": start_date,
                            "end_date": end_date,
                            "purpose": purpose
                        })
                        logger.info(f"Saved trip to long-term memory: {origin} -> {destination}")

        logger.info("Memory updated after orchestration")
