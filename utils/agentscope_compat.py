# -*- coding: utf-8 -*-
"""agentscope 1.0.x 流式解析兼容补丁。

背景：部分 OpenAI 兼容代理会在流式增量中返回 "audio": null。
agentscope 1.0.x（截至 1.0.21）的 _parse_openai_stream_response 只做了
hasattr 判断：属性存在但值为 None 时，"data" in audio 会抛
TypeError: argument of type 'NoneType' is not iterable，导致整条流式链路崩溃。

本模块提供 CompatOpenAIChatModel：仅重写流式解析入口，把传给解析器的
流包装一层，将每个增量的 audio 字段归一化为安全容器对象；
其余构造、非流式解析、工具调用等行为与 OpenAIChatModel 完全一致。
"""
from agentscope.model import OpenAIChatModel


class _NoAudio:
    """安全的 audio 占位：任何 'x in audio' 判断都返回 False。"""

    def __contains__(self, _item) -> bool:
        return False


class _NormalizedStream:
    """包装 openai AsyncStream：把每个增量的 audio 归一化后再交给解析器。"""

    def __init__(self, inner):
        self._inner = inner

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info):
        return await self._inner.__aexit__(*exc_info)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._inner.__anext__()
        self._normalize(item)
        return item

    @staticmethod
    def _normalize(item):
        # 非结构化输出时数据块就是 item 本身；
        # 结构化输出模式下真实数据块在 item.chunk 中，两者都归一化。
        targets = [item]
        chunk = getattr(item, "chunk", None)
        if chunk is not None:
            targets.append(chunk)
        for target in targets:
            for choice in getattr(target, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                audio = getattr(delta, "audio", None)
                # None 会触发 "'data' in None" 崩溃；非字典对象（如 pydantic
                # 模型）不支持 in 判断，同样会崩溃。两种情况都换成安全占位。
                if audio is None or not hasattr(audio, "__contains__"):
                    try:
                        delta.audio = _NoAudio()
                    except Exception:
                        pass  # 模型不可写时保持原状，让原异常原样抛出


class CompatOpenAIChatModel(OpenAIChatModel):
    """带 audio 判空保护的 OpenAIChatModel（兼容返回 audio:null 的第三方代理）。"""

    async def _parse_openai_stream_response(
        self,
        start_datetime,
        response,
        structured_model=None,
    ):
        wrapped = _NormalizedStream(response)
        async for chunk in super()._parse_openai_stream_response(
            start_datetime,
            wrapped,
            structured_model,
        ):
            yield chunk
