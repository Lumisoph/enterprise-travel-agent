---
name: event-collection
description: 当用户提供出发地、目的地、日期、时长或出行目的时使用。将自然语言中的行程要素提取为结构化字段，供后续行程规划使用。
---

# 行程要素收集

## 适用场景

- 用户提出新的出差或旅行需求；
- 后续规划需要结构化的城市、时间和目的信息。

## 实现

入口为 `script/agent.py` 中的 `EventCollectionAgent`。

## 输出字段

- `origin`：出发地；
- `destination`：目的地；
- `start_date`、`end_date`：开始和结束日期；
- `duration_days`：行程天数；
- `trip_purpose`：出行目的；
- `missing_info`：用户尚未提供但可能影响规划的信息。

不得猜测用户没有表达的确定事实。缺失字段使用 `null`，并写入 `missing_info`。
