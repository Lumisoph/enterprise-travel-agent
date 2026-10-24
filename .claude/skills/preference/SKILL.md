---
name: preference
description: 当用户表达或修改酒店、航司、座位、餐食、常住地等长期偏好时使用。识别本次变化是追加还是覆盖，并返回结构化更新。
---

# 用户偏好管理

## 适用场景

- “我喜欢住全季”；
- “我也常坐东航”；
- “以后座位改成靠窗”；
- “我搬到上海了”。

## 实现

- 入口：`script/agent.py` 中的 `PreferenceAgent`；
- 读取：通过 `MemoryManager` 获取当前偏好；
- 写回：由编排器或调用方统一持久化，Skill 本身只返回本次变化。

## 更新规则

- `append`：用户表达“还、也、另外”等新增选项；
- `replace`：用户表达“改成、以后只、现在是、搬到”等替换意图；
- 首次设置某个标量字段时使用 `replace`；
- 没有明确偏好时不得生成偏好。

## 输出格式

只返回严格 JSON：

```json
{
  "preferences": [
    {
      "type": "hotel_brands",
      "value": "全季",
      "action": "append"
    }
  ],
  "has_preferences": true
}
```

`action` 只能是 `append` 或 `replace`。`preferences` 只包含本次新增或变更的内容。
