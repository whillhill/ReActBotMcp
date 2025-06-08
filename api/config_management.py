from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from plugins.json_mcp import MCPConfig, MCPConfigAPI
from core.orchestrator import MCPOrchestrator
from api.models import MCPConfigResponse, MCPConfigUpdateRequest
from api.deps import get_config_api, get_orchestrator
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/mcp_config", response_model=MCPConfigResponse)
async def get_mcp_config(
    config_api: MCPConfigAPI = Depends(get_config_api)
):
    try:
        config = config_api.get_config()
        return {"mcpServers": config.get("mcpServers", {})}
    except Exception as e:
        logger.error(f"Error reading MCP configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading MCP configuration: {str(e)}")

@router.post("/update_mcp_config", response_model=Dict[str, Any])
async def update_mcp_config(
    config: MCPConfigUpdateRequest,
    config_api: MCPConfigAPI = Depends(get_config_api),
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    try:
        result = await config_api.update_config(config, orchestrator)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating MCP configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating MCP configuration: {str(e)}")

@router.post("/register_mcp_services", response_model=Dict[str, Any])
async def register_mcp_services(
    config_api: MCPConfigAPI = Depends(get_config_api),
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    try:
        result = await config_api.register_services(orchestrator)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return result
    except Exception as e:
        logger.error(f"Error registering MCP services: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error registering MCP services: {str(e)}")

@router.post("/remove_service_from_config", response_model=Dict[str, Any])
async def remove_service_from_config(
    url: str,
    service_name: str = "",
    config_api: MCPConfigAPI = Depends(get_config_api),
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    logger.info(f"Removing service from config: {url}, name: {service_name}")
    try:
        if orchestrator.registry.has_service(url):
            await orchestrator.disconnect_service(url)
        if service_name:
            success = config_api.mcp_config.remove_service(service_name)
            if success:
                return {"status": "success", "message": f"Service {service_name} removed from configuration"}
            else:
                return {"status": "warning", "message": "Service not found in configuration"}
        else:
            config = config_api.get_config()
            mcp_servers = config.get("mcpServers", {})
            service_key_to_remove = None
            for name, server_config in mcp_servers.items():
                if server_config.get("url") == url:
                    service_key_to_remove = name
                    break
            if service_key_to_remove:
                success = config_api.mcp_config.remove_service(service_key_to_remove)
                if success:
                    return {"status": "success", "message": f"Service {service_key_to_remove} removed from configuration"}
            return {"status": "warning", "message": "Service not found in configuration"}
    except Exception as e:
        logger.error(f"Error removing service from config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error removing service from configuration: {str(e)}") 
