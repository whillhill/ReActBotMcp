import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.exception_handlers import validation_exception_handler
from api.llm_agent import router as llm_agent_router
from api.service_management import router as service_management_router
from api.config_management import router as config_management_router
from api.tool_catalog import router as tool_catalog_router
from api.deps import app_state
from plugins.json_mcp import MCPConfig, MCPConfigAPI
from core.registry import ServiceRegistry
from core.orchestrator import MCPOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("mcp_service.log")])
logger = logging.getLogger(__name__)

async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing components...")
    config_dir = os.path.dirname(__file__)
    mcp_config_handler = MCPConfig(os.path.join(config_dir, "mcp.json"))
    config = mcp_config_handler.load_config()
    config_api = MCPConfigAPI(os.path.join(config_dir, "mcp.json"))
    app_state["config_api"] = config_api
    registry = ServiceRegistry()
    orchestrator = MCPOrchestrator(config=config, registry=registry)
    await orchestrator.setup()
    await orchestrator.start_monitoring()
    logger.info("Registering services from mcp.json...")
    register_result = await config_api.register_services(orchestrator)
    if register_result.get("status") == "success":
        logger.info(f"Services registered: {register_result.get('message')}")
    else:
        logger.error(f"Service registration failed: {register_result.get('message')}")
    app_state["orchestrator"] = orchestrator
    app_state["registry"] = registry
    app_state["mcp_config"] = mcp_config_handler
    logger.info("Components initialized and background tasks started.")
    yield
    logger.info("Application shutdown: Cleaning up resources...")
    orch = app_state.get("orchestrator")
    if orch:
        await orch.cleanup()
    app_state.clear()
    logger.info("Application shutdown complete.")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(llm_agent_router)
app.include_router(service_management_router)
app.include_router(config_management_router)
app.include_router(tool_catalog_router)

# 注册异常处理
from fastapi.exceptions import RequestValidationError
app.add_exception_handler(RequestValidationError, validation_exception_handler) 
