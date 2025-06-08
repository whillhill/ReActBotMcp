from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from core.registry import ServiceRegistry
from core.orchestrator import MCPOrchestrator
from api.models import RegisterRequest, ServiceInfoRequest
from api.deps import get_orchestrator, get_registry
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register", response_model=Dict[str, str])
async def register_service_endpoint(
    payload: RegisterRequest,
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    server_url_str = str(payload.url)
    service_name = payload.name or server_url_str.split('/')[-2]
    logger.info(f"Received registration request, target URL: {server_url_str}, service name: {service_name}")
    try:
        success, message = await orchestrator.connect_service(server_url_str, service_name)
        if success:
            logger.info(f"Service {service_name} ({server_url_str}) registered successfully: {message}")
            return {"status": "success", "message": message}
        else:
            logger.error(f"Service {service_name} ({server_url_str}) registration failed: {message}")
            status_code = 500
            is_connection_issue = False
            if "502 Bad Gateway" in message: status_code = 502; is_connection_issue = True
            elif "Connection failed" in message or "Network connection error" in message: status_code = 502; is_connection_issue = True
            if is_connection_issue:
                logger.info(f"Adding service {service_name} ({server_url_str}) to auto-reconnect list.")
                orchestrator.pending_reconnection.add(server_url_str)
            raise HTTPException(status_code=status_code, detail=message)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unknown error processing registration request (URL: {server_url_str}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred while processing the registration request.")

@router.get("/health", response_model=Dict[str, Any])
async def get_health_status(
    registry: ServiceRegistry = Depends(get_registry),
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    service_statuses = registry.get_registered_services_details()
    for status in service_statuses:
        is_healthy = await orchestrator.is_service_healthy(status["url"])
        status["status"] = "healthy" if is_healthy else "unhealthy"
    return {
        "orchestrator_status": "running",
        "active_services": registry.get_session_count(),
        "total_tools": registry.get_tool_count(),
        "services": service_statuses
    }

@router.get("/service_info", response_model=Dict[str, Any])
async def get_service_info(
    url: str,
    registry: ServiceRegistry = Depends(get_registry),
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
):
    if not registry.has_service(url):
        raise HTTPException(status_code=404, detail=f"Service not found: {url}")
    service_info = registry.get_service_info(url)
    is_healthy = await orchestrator.is_service_healthy(url)
    service_info["status"] = "healthy" if is_healthy else "unhealthy"
    tools = registry.get_tools_for_service(url)
    service_info["tools"] = [{"name": name, "description": tool.get("function", {}).get("description", "")} for name, tool in tools]
    return service_info

@router.get("/services")
async def list_services(
    registry: ServiceRegistry = Depends(get_registry)
):
    services = registry.get_registered_services_details()
    return {"services": services} 
