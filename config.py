"""
企业差旅多智能体系统配置。
"""

# LLM Configuration
LLM_CONFIG = {
    "api_key": "API_KEY",
    "model_name": "Model_Name",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "temperature": 0.7,
    "max_tokens": 8192,
}

# System Configuration
SYSTEM_CONFIG = {
    "enable_llm": True,  # Set to True to use LLM (recommended), False for rule-based
    "log_level": "INFO",
    "max_retries": 3,
    "timeout": 60,  # Increased timeout for better stability
}

# RAG 知识库：嵌入模型（本地路径，无需连 HuggingFace）
RAG_CONFIG = {
    "embedding_model": "data/models/bge-small-zh-v1.5",
}

# 连接与可用性：重试、熔断、健康检查
RESILIENCE_CONFIG = {
    "max_retries": 3,              # 单次请求最大重试次数（与 SYSTEM_CONFIG 对齐）
    "retry_base_delay_sec": 1.0,   # 重试退避基数（秒）
    "retry_max_delay_sec": 30.0,   # 重试退避上限（秒）
    "circuit_failure_threshold": 5, # 连续失败多少次后熔断
    "circuit_recovery_timeout_sec": 60.0,  # 熔断后多少秒进入半开
    "circuit_half_open_successes": 2,      # 半开状态下连续成功多少次后关闭
    "health_check_timeout_sec": 10.0,      # 健康检查请求超时（秒）
}

# 本机 .env 配置覆盖：密钥与模型名不写入代码仓库。
# 约定与 experiments/model.py 一致：TRAVEL_LLM_API_KEY / TRAVEL_LLM_MODEL / TRAVEL_LLM_BASE_URL。
# .env 不存在或缺少键时保持上面的占位值（供无密钥环境做导入/离线测试）。
def _load_env_overrides():
    from pathlib import Path
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    values = {}
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    if values.get("TRAVEL_LLM_API_KEY"):
        LLM_CONFIG["api_key"] = values["TRAVEL_LLM_API_KEY"]
    if values.get("TRAVEL_LLM_MODEL"):
        LLM_CONFIG["model_name"] = values["TRAVEL_LLM_MODEL"]
    if values.get("TRAVEL_LLM_BASE_URL"):
        LLM_CONFIG["base_url"] = values["TRAVEL_LLM_BASE_URL"]

_load_env_overrides()
