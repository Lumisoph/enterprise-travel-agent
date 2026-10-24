# -*- coding: utf-8 -*-
"""
agentscope 轻量测试桩

当前开发环境未安装 agentscope（真实环境通过 requirements.txt 安装）。
本模块把 agentscope.agent / agentscope.message / agentscope.model 注入 sys.modules，
使产品代码（协调器、真实 Skill）可以在离线测试中运行。
只用于测试，不参与任何实验测量，不代表真实框架行为。
"""
import sys
import types


class AgentBase:
    """桩基类：兼容产品代码 super().__init__() 调用。"""
    def __init__(self, *args, **kwargs):
        pass


class Msg:
    """桩消息：兼容 name / content / role 属性。"""
    def __init__(self, name, content, role):
        self.name = name
        self.content = content
        self.role = role


class OpenAIChatModel:
    """桩模型类：仅占位，真实调用在测试中用假模型替代。"""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("测试中请使用假模型替代 OpenAIChatModel")


def install():
    """安装桩模块；重复调用无副作用。"""
    if "agentscope" in sys.modules:
        return
    agentscope = types.ModuleType("agentscope")
    agent_mod = types.ModuleType("agentscope.agent")
    agent_mod.AgentBase = AgentBase
    msg_mod = types.ModuleType("agentscope.message")
    msg_mod.Msg = Msg
    model_mod = types.ModuleType("agentscope.model")
    model_mod.OpenAIChatModel = OpenAIChatModel
    agentscope.agent = agent_mod
    agentscope.message = msg_mod
    agentscope.model = model_mod
    sys.modules.update({
        "agentscope": agentscope,
        "agentscope.agent": agent_mod,
        "agentscope.message": msg_mod,
        "agentscope.model": model_mod,
    })
