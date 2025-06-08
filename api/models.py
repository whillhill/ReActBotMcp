from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, Optional, Literal

class UnifiedQueryRequest(BaseModel):
    query: str
    mode: Optional[Literal["standard", "react"]] = "react"
    stream_type: Optional[Literal[None, "step", "token"]] = None
    include_trace: Optional[bool] = False

class RegisterRequest(BaseModel):
    url: HttpUrl
    name: str = ""

class ServiceInfoRequest(BaseModel):
    url: str

class MCPConfigResponse(BaseModel):
    mcpServers: Dict[str, Dict[str, Any]]

class MCPConfigUpdateRequest(BaseModel):
    mcpServers: Dict[str, Dict[str, Any]] 
