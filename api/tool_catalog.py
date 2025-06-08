from fastapi import APIRouter, Depends
from core.registry import ServiceRegistry
from api.deps import get_registry

router = APIRouter()

@router.get("/tools")
async def list_tools(
    registry: ServiceRegistry = Depends(get_registry)
):
    tools = registry.get_all_tools()
    return {"tools": tools} 
