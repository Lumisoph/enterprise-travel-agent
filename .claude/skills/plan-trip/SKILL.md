---
name: plan-trip
description: 当用户要求规划出差、旅行或路线时使用。综合已收集的日期、城市、政策、偏好和实时信息，输出结构化行程方案。
---

# 行程规划

## 适用场景

- 用户要求规划行程或安排路线；
- 用户已经提供目的地、日期等差旅信息；
- 前序 Skill 已获得政策、偏好或天气，需要统一形成方案。

## 实现

入口为 `script/agent.py` 中的 `ItineraryPlanningAgent`。输入包含用户需求、用户偏好以及其他 Skill 的结构化结果。

## 规划原则

1. 优先遵守明确的时间、预算和政策约束；
2. 已提供天气信息时，给出对应日期的具体应对动作；
3. 不虚构实时票价、库存、营业状态或用户偏好；
4. 信息不完整时仍可给出有用的初步方案，同时在 `missing_info` 中列出待确认项；
5. 每日活动数量应考虑交通、用餐和休息时间。

## 输出格式

只返回严格 JSON：

```json
{
  "itinerary": {
    "title": "行程标题",
    "duration": "3天",
    "route": "北京 → 杭州",
    "daily_plans": [
      {
        "day": 1,
        "date": "2026-11-02",
        "city": "杭州",
        "theme": "抵达与会议",
        "activities": [],
        "meals": {}
      }
    ],
    "notes": [],
    "estimated_budget": "待确认"
  },
  "planning_complete": true,
  "missing_info": []
}
```
