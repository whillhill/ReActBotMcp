import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
基础客户端接口定义

本模块定义了所有MCP客户端实现必须遵循的基础接口。
这些接口确保不同的客户端实现（如官方Client、增强型Client等）
可以互换使用，同时保持一致的API。
"""

import abc
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, Tuple

class BaseClient(abc.ABC):
    """
    MCP客户端基础接口
    
    定义了与MCP服务交互的基本方法，所有客户端实现都应该遵循这个接口。
    """
    
    @abc.abstractmethod
    async def __aenter__(self):
        """
        实现异步上下文管理器的进入方法
        
        Returns:
            客户端实例
        """
        pass
    
    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        实现异步上下文管理器的退出方法
        """
        pass
    
    @abc.abstractmethod
    async def list_tools(self) -> List[Any]:
        """
        获取可用工具列表
        
        Returns:
            工具定义列表
        """
        pass
    
    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用指定的工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        pass
    
    @abc.abstractmethod
    async def ping(self) -> bool:
        """
        检查服务连接状态
        
        Returns:
            服务是否可用
        """
        pass
    
    @abc.abstractmethod
    async def is_service_healthy(self) -> bool:
        """
        检查服务健康状态
        
        Returns:
            服务是否健康
        """
        pass

class EnhancedClientInterface(BaseClient):
    """
    增强型MCP客户端接口
    
    在基础接口之上，添加了更高级的功能，如查询处理、流式响应等。
    """
    
    @abc.abstractmethod
    async def process_query(self, query: str) -> str:
        """
        使用标准方法处理用户查询
        
        Args:
            query: 用户查询字符串
            
        Returns:
            处理结果
        """
        pass
    
    @abc.abstractmethod
    async def process_query_with_react(self, query: str) -> str:
        """
        使用ReAct模式处理用户查询
        
        Args:
            query: 用户查询字符串
            
        Returns:
            处理结果
        """
        pass
    
    @abc.abstractmethod
    async def process_query_with_trace(self, query: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        使用ReAct模式处理用户查询并返回执行跟踪
        
        Args:
            query: 用户查询字符串
            
        Returns:
            (处理结果, 执行跟踪)
        """
        pass
    
    @abc.abstractmethod
    async def stream_process_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户查询（步骤级）
        
        Args:
            query: 用户查询字符串
            
        Returns:
            流式响应生成器
        """
        pass
    
    @abc.abstractmethod
    async def stream_process_query_token(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户查询（令牌级）
        
        Args:
            query: 用户查询字符串
            
        Returns:
            流式响应生成器
        """
        pass
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
    async def start_monitoring(self) -> None:
        """
        启动健康监控
        """
        pass
    
    @abc.abstractmethod
    async def stop_monitoring(self) -> None:
        """
        停止健康监控
        """
        pass
    
    @abc.abstractmethod
    async def cleanup(self) -> None:
        """
        清理资源
        """
        pass 
