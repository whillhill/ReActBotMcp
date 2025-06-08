from typing import Dict, Any
from core.orchestrator import MCPOrchestrator
from core.registry import ServiceRegistry
from plugins.json_mcp import MCPConfigAPI
from fastapi import HTTPException

# 全局应用状态
app_state: Dict[str, Any] = {}

def get_orchestrator() -> MCPOrchestrator:
    orchestrator = app_state.get("orchestrator")
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not ready (Orchestrator not initialized)")
    return orchestrator

def get_registry() -> ServiceRegistry:
    registry = app_state.get("registry")
    if registry is None:
        raise HTTPException(status_code=503, detail="Service not ready (Registry not initialized)")
    return registry

def get_config_api() -> MCPConfigAPI:
    config_api = app_state.get("config_api")
    if config_api is None:
        raise HTTPException(status_code=503, detail="Service not ready (Config API not initialized)")
    return config_api 
