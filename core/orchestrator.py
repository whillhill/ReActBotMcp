import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
MCP服务编排器

该模块提供了MCPOrchestrator类，用于管理MCP服务的连接、工具调用和查询处理。
它是FastAPI应用程序的核心组件，负责协调客户端和服务之间的交互。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, AsyncGenerator
from datetime import datetime, timedelta
from urllib.parse import urljoin

from core.registry import ServiceRegistry
from fastmcp import Client
from fastmcp.client.transports import MCPConfigTransport
from plugins.json_mcp import MCPConfig

logger = logging.getLogger(__name__)

class MCPOrchestrator:
    """
    MCP服务编排器
    
    负责管理服务连接、工具调用和查询处理。
    """
    
    def __init__(self, config: Dict[str, Any], registry: ServiceRegistry):
        """
        初始化MCP编排器
        
        Args:
            config: 配置字典
            registry: 服务注册表实例
        """
        self.config = config
        self.registry = registry
        self.clients: Dict[str, Client] = {}  # key为mcpServers的服务名
        self.pending_reconnection: Set[str] = set()
        self.react_agent = None
        
        # 从配置中获取心跳和重连设置
        timing_config = config.get("timing", {})
        self.heartbeat_interval = timedelta(seconds=int(timing_config.get("heartbeat_interval_seconds", 60)))
        self.heartbeat_timeout = timedelta(seconds=int(timing_config.get("heartbeat_timeout_seconds", 180)))
        self.reconnection_interval = timedelta(seconds=int(timing_config.get("reconnection_interval_seconds", 60)))
        self.http_timeout = int(timing_config.get("http_timeout_seconds", 10))
        
        # 监控任务
        self.heartbeat_task = None
        self.reconnection_task = None
        self.mcp_config = MCPConfig()
    
    async def setup(self):
        """初始化编排器资源"""
        logger.info("Setting up MCP Orchestrator...")
        await self.load_from_config()
    
    async def load_from_config(self):
        """从mcp.json加载所有服务，使用MCPConfigTransport批量连接"""
        logger.info("Loading all MCP services from mcp.json via MCPConfigTransport...")
        config_dict = self.mcp_config.load_config()
        transport = MCPConfigTransport(config_dict)
        client = Client(transport)
        self.clients.clear()
        self.registry.clear()
        for name, server in config_dict.get("mcpServers", {}).items():
            try:
                self.clients[name] = client
                self.registry.add_service(name, client, [], name)
                logger.info(f"Registered service: {name} -> {server.get('url', name)}")
            except Exception as e:
                logger.error(f"Failed to register service {name}: {e}")
    
    async def start_monitoring(self):
        """启动后台健康检查和重连监视器"""
        logger.info("Starting monitoring tasks...")
        
        # 启动心跳监视器
        if self.heartbeat_task is None or self.heartbeat_task.done():
            logger.info(f"Starting heartbeat monitor. Interval: {self.heartbeat_interval.total_seconds()}s")
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # 启动重连监视器
        if self.reconnection_task is None or self.reconnection_task.done():
            logger.info(f"Starting reconnection monitor. Interval: {self.reconnection_interval.total_seconds()}s")
            self.reconnection_task = asyncio.create_task(self._reconnection_loop())
    
    async def _heartbeat_loop(self):
        """后台循环，用于定期健康检查"""
        while True:
            await asyncio.sleep(self.heartbeat_interval.total_seconds())
            await self._check_services_health()
    
    async def _check_services_health(self):
        """检查所有服务的健康状态"""
        logger.debug("Running periodic health check for all services...")
        for name in self.clients:
            try:
                is_healthy = await self.is_service_healthy(name)
                if is_healthy:
                    logger.debug(f"Health check SUCCESS for: {name}")
                    self.registry.update_service_health(name)
                else:
                    logger.warning(f"Health check FAILED for {name}")
                    self.pending_reconnection.add(name)
            except Exception as e:
                logger.warning(f"Health check error for {name}: {e}")
                self.pending_reconnection.add(name)
    
    async def _reconnection_loop(self):
        """定期尝试重新连接服务的后台循环"""
        while True:
            await asyncio.sleep(self.reconnection_interval.total_seconds())
            await self._attempt_reconnections()
    
    async def _attempt_reconnections(self):
        """尝试重新连接所有待重连的服务"""
        if not self.pending_reconnection:
            return  # 如果没有待重连的服务，跳过
        
        # 创建副本以避免迭代过程中修改集合的问题
        names_to_retry = list(self.pending_reconnection)
        logger.info(f"Attempting to reconnect {len(names_to_retry)} service(s): {names_to_retry}")
        
        for name in names_to_retry:
            try:
                # 尝试重新连接
                success, message = await self.connect_service(name, name)
                if success:
                    logger.info(f"Reconnection successful for: {name}")
                    self.pending_reconnection.discard(name)
                else:
                    logger.warning(f"Reconnection attempt failed for {name}: {message}")
                    # 保持name在pending_reconnection中，等待下一个周期
            except Exception as e:
                logger.warning(f"Reconnection attempt failed for {name}: {e}")
    
    async def connect_service(self, url: str, name: str = "") -> Tuple[bool, str]:
        """添加服务到mcp.json并刷新所有连接"""
        logger.info(f"Registering new service: {url}, name: {name}")
        if not name:
            name = url
        ok = self.mcp_config.add_service({"name": name, "url": url})
        if ok:
            await self.load_from_config()
            return True, f"Service {name} registered and all services refreshed."
        else:
            return False, f"Failed to add service {name} to mcp.json."
    
    async def disconnect_service(self, url: str) -> bool:
        """从mcp.json移除服务并刷新所有连接"""
        logger.info(f"Removing service: {url}")
        config = self.mcp_config.load_config()
        servers = config.get("mcpServers", {})
        name_to_remove = None
        for name, server in servers.items():
            if server.get("url") == url or name == url:
                name_to_remove = name
                break
        if name_to_remove:
            ok = self.mcp_config.remove_service(name_to_remove)
            if ok:
                await self.load_from_config()
                return True
        logger.warning(f"Service {url} not found in mcp.json.")
        return False
    
    async def refresh_services(self):
        """手动刷新所有服务连接（重新加载mcp.json）"""
        await self.load_from_config()
    
    async def is_service_healthy(self, name: str) -> bool:
        """
        检查服务是否健康
        
        Args:
            name: 服务名
            
        Returns:
            服务是否健康
        """
        client = self.clients.get(name)
        if not client:
            return False
        
        try:
            return await client.is_service_healthy()
        except Exception as e:
            logger.warning(f"Health check failed for {name}: {e}")
            return False
    
    async def process_unified_query(
        self, 
        query: str, 
        mode: str = "react",
        stream_type: Optional[str] = None,
        include_trace: bool = False
    ) -> Union[str, Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        统一的查询处理方法
        
        Args:
            query: 用户查询字符串
            mode: 处理模式，'standard' 或 'react'
            stream_type: 流式类型，None(非流式)、'step'(步骤级)或'token'(令牌级)
            include_trace: 是否包含执行轨迹
            
        Returns:
            根据参数返回不同类型的结果
        """
        logger.info(f"Processing unified query: '{query[:50]}...', mode: {mode}, stream: {stream_type}")
        
        # 检查是否有ReAct代理
        if mode == "react" and not self.react_agent:
            # 找到具有ReAct功能的客户端
            for client in self.clients.values():
                if client.react_agent:
                    self.react_agent = client.react_agent
                    # 确保react_agent有访问registry的权限
                    self.react_agent.registry = self.registry
                    logger.info("Found and using ReAct agent from a connected service.")
                    break
            
            if not self.react_agent:
                logger.warning("No ReAct agent available. Falling back to standard mode.")
                mode = "standard"
        
        # 选择客户端进行处理
        if self.clients:
            # 现在使用第一个客户端
            # 在更复杂的实现中，我们可能希望根据功能选择
            client = next(iter(self.clients.values()))
            
            # 使用选定的客户端处理查询
            return await client.process_unified_query(
                query=query,
                mode=mode,
                stream_type=stream_type,
                include_trace=include_trace
            )
        else:
            logger.error("No clients available for processing query")
            return "Error: No services connected. Please connect at least one MCP service."
    
    async def stream_process_query(self, query: str):
        """
        流式处理查询（步骤级）
        
        Args:
            query: 用户查询字符串
            
        Yields:
            步骤级响应流
        """
        logger.info(f"Stream processing query (step-level): '{query[:50]}...'")
        
        if not self.clients:
            yield {
                "thinking_step": None,
                "is_final": True,
                "result": "Error: No services connected. Please connect at least one MCP service."
            }
            return
        
        # 现在使用第一个客户端
        client = next(iter(self.clients.values()))
        
        # 使用客户端的流式方法
        async for response in client.stream_process_query(query):
            yield response
    
    async def stream_process_query_token(self, query: str):
        """
        流式处理查询（令牌级）
        
        Args:
            query: 用户查询字符串
            
        Yields:
            令牌级响应流
        """
        logger.info(f"Stream processing query (token-level): '{query[:50]}...'")
        
        if not self.clients:
            yield {
                "token_chunk": None,
                "is_final": True,
                "result": "Error: No services connected. Please connect at least one MCP service."
            }
            return
        
        # 现在使用第一个客户端
        client = next(iter(self.clients.values()))
        
        # 使用客户端的令牌流式方法
        async for response in client.stream_process_query_token(query):
            yield response
    
    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up orchestrator resources...")
        
        # 停止监控任务
        tasks_to_stop = [self.heartbeat_task, self.reconnection_task]
        task_names = ["Heartbeat", "Reconnection"]
        for i, task in enumerate(tasks_to_stop):
            name = task_names[i]
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"{name} monitor task cancelled.")
                except Exception as e:
                    logger.error(f"Error during {name} task cancellation: {e}", exc_info=True)
        
        # 断开所有服务连接
        for name in list(self.clients.keys()):
            await self.disconnect_service(name)
        
        logger.info("Orchestrator cleanup finished.") 
