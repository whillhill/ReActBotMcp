import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
客户端适配器，用于桥接官方Client接口和我们的功能
"""

import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Union, Tuple
from fastmcp import Client

from core.base_client import BaseClient

logger = logging.getLogger(__name__)

class ClientAdapter(BaseClient):
    """
    适配官方Client接口以支持我们的功能
    
    这个类包装了官方Client，提供了标准的BaseClient接口实现，
    同时添加了工具会话管理、健康检查和错误处理等功能。
    """
    
    def __init__(self, client: Client):
        """
        初始化客户端适配器
        
        Args:
            client: 官方Client实例
        """
        self.client = client
        self.tool_sessions = {}  # 工具名称到会话的映射
        self._connected = False  # 连接状态跟踪
        
    async def __aenter__(self):
        """
        实现异步上下文管理器的进入方法
        
        Returns:
            ClientAdapter实例
        """
        if not self._connected:
            logger.debug("Entering ClientAdapter context, connecting client...")
            await self.client.__aenter__()
            self._connected = True
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        实现异步上下文管理器的退出方法
        """
        if self._connected:
            logger.debug("Exiting ClientAdapter context, disconnecting client...")
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
            self._connected = False
    
    async def _ensure_connected(self):
        """
        确保客户端已连接
        
        如果客户端未连接，则尝试连接
        """
        if not self._connected:
            logger.debug("Client not connected, connecting now...")
            await self.__aenter__()
    
    async def get_session_for_tool(self, tool_name: str) -> Optional[Client]:
        """
        获取工具对应的会话
        
        Args:
            tool_name: 工具名称
            
        Returns:
            处理该工具的Client实例
        """
        # 在当前实现中，所有工具都由同一个Client处理
        await self._ensure_connected()
        return self.client
    
    async def list_tools(self) -> List[Any]:
        """
        获取所有工具
        
        实现BaseClient接口方法，直接使用官方Client的list_tools方法
        
        Returns:
            工具定义列表
        """
        try:
            await self._ensure_connected()
            return await self.client.list_tools()
        except Exception as e:
            logger.error(f"Error listing tools: {e}", exc_info=True)
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用工具
        
        实现BaseClient接口方法，使用官方Client的call_tool方法
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        try:
            await self._ensure_connected()
            # 直接使用官方接口
            return await self.client.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def ping(self) -> bool:
        """
        检查服务连接状态
        
        实现BaseClient接口方法，使用官方Client的ping方法
        
        Returns:
            服务是否可用
        """
        try:
            await self._ensure_connected()
            await self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False
    
    async def is_service_healthy(self) -> bool:
        """
        检查服务是否健康
        
        实现BaseClient接口方法，使用ping方法检查健康状态
        
        Returns:
            服务健康状态
        """
        return await self.ping()
    
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的格式化定义
        
        Returns:
            工具定义列表，格式化为LLM可用的格式
        """
        try:
            await self._ensure_connected()
            # 获取工具列表
            tools = await self.list_tools()
            
            # 处理工具定义
            processed_tools = []
            for tool in tools:
                # 获取工具参数schema
                parameters = getattr(tool, 'inputSchema', {}) or {}
                
                # 确保参数格式正确
                if not isinstance(parameters, dict) or parameters.get("type") != "object":
                    parameters = {
                        "type": "object", 
                        "properties": parameters, 
                        "required": list(parameters.keys()) if isinstance(parameters, dict) else []
                    }
                
                # 创建LLM工具定义
                tool_definition = {
                    "type": "function", 
                    "function": {
                        "name": tool.name, 
                        "description": getattr(tool, 'description', f"Tool {tool.name}"), 
                        "parameters": parameters
                    }
                }
                processed_tools.append(tool_definition)
                
            return processed_tools
        except Exception as e:
            logger.error(f"Error getting tools: {e}", exc_info=True)
            return []
    
    async def update_service_health(self) -> None:
        """
        更新服务健康状态
        """
        # 在当前实现中，我们不需要特别的健康状态更新逻辑
        pass 
