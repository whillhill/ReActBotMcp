import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Configuration Constants (default values) ---
HEARTBEAT_INTERVAL_SECONDS = 60
HEARTBEAT_TIMEOUT_SECONDS = 180
HTTP_TIMEOUT_SECONDS = 10
RECONNECTION_INTERVAL_SECONDS = 60
REACT_MAX_ITERATIONS = 5
REACT_ENABLE_TRACE = False
STREAMABLE_HTTP_ENDPOINT = "/mcp"

@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    api_key: str = ""
    model: str = ""
    base_url: Optional[str] = None

def load_llm_config() -> LLMConfig:
    """从环境变量加载LLM配置（仅支持openai兼容接口）"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    provider = "openai_compatible"
    if not api_key:
        logger.warning("OPENAI_API_KEY not set in environment.")
    if not model:
        logger.warning("OPENAI_MODEL not set in environment.")
    return LLMConfig(provider=provider, api_key=api_key, model=model, base_url=base_url)

def _get_env_int(var: str, default: int) -> int:
    try:
        return int(os.environ.get(var, default))
    except Exception:
        logger.warning(f"环境变量{var}格式错误，使用默认值{default}")
        return default

def _get_env_bool(var: str, default: bool) -> bool:
    val = os.environ.get(var)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

def load_app_config() -> Dict[str, Any]:
    """从环境变量加载全局配置"""
    config_data = {
        "heartbeat_interval": _get_env_int("HEARTBEAT_INTERVAL_SECONDS", HEARTBEAT_INTERVAL_SECONDS),
        "heartbeat_timeout": _get_env_int("HEARTBEAT_TIMEOUT_SECONDS", HEARTBEAT_TIMEOUT_SECONDS),
        "http_timeout": _get_env_int("HTTP_TIMEOUT_SECONDS", HTTP_TIMEOUT_SECONDS),
        "reconnection_interval": _get_env_int("RECONNECTION_INTERVAL_SECONDS", RECONNECTION_INTERVAL_SECONDS),
        "react_max_iterations": _get_env_int("REACT_MAX_ITERATIONS", REACT_MAX_ITERATIONS),
        "react_enable_trace": _get_env_bool("REACT_ENABLE_TRACE", REACT_ENABLE_TRACE),
        "streamable_http_endpoint": os.environ.get("STREAMABLE_HTTP_ENDPOINT", STREAMABLE_HTTP_ENDPOINT),
    }
    # 加载LLM配置
    config_data["llm_config"] = load_llm_config()
    logger.info(f"Loaded configuration from environment: {config_data}")
    return config_data
