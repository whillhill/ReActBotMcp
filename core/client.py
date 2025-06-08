import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import json
import httpx
import logging
from datetime import datetime, timedelta
from contextlib import AsyncExitStack
from urllib.parse import urljoin
from typing import Dict, List, Optional, Any, Tuple, Set, Union, AsyncGenerator

from core.base_client import EnhancedClientInterface
from core.client_adapter import ClientAdapter
from core.registry import ServiceRegistry
from plugins.llm_factory import create_llm_client
from plugins.react_agent import ReActAgent
from core.transport import StreamableHTTPConfig, StreamableHTTPTransport
from fastmcp import Client

logger = logging.getLogger(__name__)

class EnhancedClient(EnhancedClientInterface):
    """扩展官方Client类，添加额外功能如ReAct代理和流式处理"""
    
    def __init__(self, server_url: str, **kwargs):
        """初始化增强客户端
        
        Args:
            server_url: 服务器URL
            **kwargs: 额外配置参数，包括：
                - config: 配置字典
                - heartbeat_interval: 心跳间隔（秒）
                - heartbeat_timeout: 心跳超时（秒）
                - reconnection_interval: 重连间隔（秒）
                - http_timeout: HTTP超时（秒）
        """
        # 创建官方Client实例
        self.official_client = Client(server_url)
        
        # 使用ClientAdapter包装官方Client
        self.adapter = ClientAdapter(self.official_client)
        
        # 保存基础URL
        self.base_url = server_url
        
        # 配置参数
        self.config = kwargs.get("config", {})
        self.llm_client = None
        self.react_agent = None
        
        # 心跳和重连配置
        self.heartbeat_interval = timedelta(seconds=int(kwargs.get("heartbeat_interval", 60)))
        self.heartbeat_timeout = timedelta(seconds=int(kwargs.get("heartbeat_timeout", 180)))
        self.reconnection_interval = timedelta(seconds=int(kwargs.get("reconnection_interval", 60)))
        self.http_timeout = int(kwargs.get("http_timeout", 10))
        
        # 心跳和重连任务
        self.heartbeat_task = None
        self.reconnection_task = None
        self.pending_reconnection = set()
        
        # 初始化LLM客户端（如果配置了）
        llm_config = self.config.get("llm_config")
        if llm_config:
            self._initialize_llm_client(llm_config)
    
    def _initialize_llm_client(self, llm_config):
        """初始化LLM客户端和ReAct代理"""
        try:
            # 检查配置完整性
            if not llm_config.api_key or not llm_config.model:
                missing = []
                if not llm_config.api_key: missing.append("api_key")
                if not llm_config.model: missing.append("model")
                logger.error(f"LLM配置不完整，缺少必要参数: {', '.join(missing)}")
                logger.error(f"当前配置: provider={llm_config.provider}, model={llm_config.model}, base_url={llm_config.base_url}")
                logger.error("ReAct Agent将无法初始化，流式功能将不可用")
                return
            elif not llm_config.provider:
                logger.error("LLM配置缺少provider字段")
                logger.error("ReAct Agent将无法初始化，流式功能将不可用")
                return
            
            self.llm_client = create_llm_client(llm_config)
            if self.llm_client:
                logger.info(f"{llm_config.provider.capitalize()} Client initialized with model {llm_config.model}.")
                # Initialize ReAct agent if LLM client is available
                try:
                    # 创建ReActAgent时传入registry=None参数，将在orchestrator中设置
                    self.react_agent = ReActAgent(self.llm_client, self, self.config, registry=None)
                    self.react_agent.is_service_healthy = self.is_service_healthy
                    logger.info("ReAct Agent initialized successfully.")
                except Exception as react_err:
                    logger.error(f"ReAct Agent初始化失败: {react_err}", exc_info=True)
                    self.react_agent = None
            else:
                logger.warning(f"LLM Client初始化失败，provider={llm_config.provider}，model={llm_config.model}")
        except Exception as e:
            logger.error(f"创建LLM客户端时发生错误: {e}", exc_info=True)
    
    # 添加异步上下文管理器支持
    async def __aenter__(self):
        """
        实现异步上下文管理器的进入方法
        
        Returns:
            EnhancedClient实例
        """
        # 使用adapter的上下文管理器
        await self.adapter.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        实现异步上下文管理器的退出方法
        """
        # 使用adapter的上下文管理器
        await self.adapter.__aexit__(exc_type, exc_val, exc_tb)
    
    # 实现BaseClient接口方法，委托给adapter
    
    async def list_tools(self) -> List[Any]:
        """获取可用工具列表，委托给adapter"""
        return await self.adapter.list_tools()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """调用指定的工具，委托给adapter"""
        return await self.adapter.call_tool(tool_name, arguments)
    
    async def ping(self) -> bool:
        """检查服务连接状态，委托给adapter"""
        return await self.adapter.ping()
    
    async def is_service_healthy(self) -> bool:
        """检查服务健康状态，委托给adapter"""
        try:
            await self.ping()
            return True
        except Exception:
            return False
    
    # 实现EnhancedClientInterface接口方法
    
    async def start_monitoring(self):
        """启动后台健康检查和重连监视器"""
        # 启动心跳监视器
        if self.heartbeat_task is None or self.heartbeat_task.done():
            logger.info(f"Starting heartbeat monitor. Interval: {self.heartbeat_interval.total_seconds()}s")
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # 启动重连监视器
        if self.reconnection_task is None or self.reconnection_task.done():
            logger.info(f"Starting reconnection monitor. Interval: {self.reconnection_interval.total_seconds()}s")
            self.reconnection_task = asyncio.create_task(self._reconnection_loop())

    async def stop_monitoring(self):
        """停止后台健康检查和重连监视器"""
        tasks_to_stop = [self.heartbeat_task, self.reconnection_task]
        task_names = ["Heartbeat", "Reconnection"]
        for i, task in enumerate(tasks_to_stop):
            name = task_names[i]
            if task and not task.done():
                task.cancel()
                try: await task
                except asyncio.CancelledError: logger.info(f"{name} monitor task cancelled.")
                except Exception as e: logger.error(f"Error during {name} task cancellation: {e}", exc_info=True)
        self.heartbeat_task = None
        self.reconnection_task = None
    
    async def _heartbeat_loop(self):
        """后台循环，用于定期健康检查"""
        while True:
            await asyncio.sleep(self.heartbeat_interval.total_seconds())
            await self._check_service_health()
    
    async def _check_service_health(self):
        """检查服务健康状态"""
        logger.debug("Running periodic health check...")
        try:
            await self.ping()
            logger.debug(f"Health check SUCCESS for: {self.base_url}")
        except Exception as e:
            logger.warning(f"Health check FAILED for {self.base_url}: {e}")
            self.pending_reconnection.add(self.base_url)
    
    async def _reconnection_loop(self):
        """定期尝试重新连接服务的后台循环"""
        while True:
            await asyncio.sleep(self.reconnection_interval.total_seconds())
            await self._attempt_reconnections()
    
    async def _attempt_reconnections(self):
        """尝试重新连接一次所有待重连的服务"""
        if not self.pending_reconnection:
            return  # 如果没有待重连的服务，跳过
        
        # 创建副本以避免迭代过程中修改集合的问题
        urls_to_retry = list(self.pending_reconnection)
        logger.info(f"Attempting to reconnect {len(urls_to_retry)} service(s): {urls_to_retry}")
        
        for url in urls_to_retry:
            try:
                await self.ping()
                logger.info(f"Reconnection successful for: {url}")
                self.pending_reconnection.discard(url)
            except Exception as e:
                logger.warning(f"Reconnection attempt failed for {url}: {e}")
                # 保持URL在self.pending_reconnection中，等待下一个周期
    
    async def process_unified_query(
        self, 
        query: str, 
        mode: str = "react",
        stream_type: Optional[str] = None,
        include_trace: bool = False
    ) -> Union[str, Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        统一的查询处理方法，支持不同的模式和流式选项
        
        Args:
            query: 用户查询字符串
            mode: 处理模式，'standard' 或 'react'
            stream_type: 流式类型，None(非流式)、'step'(步骤级流式)或'token'(令牌级流式)
            include_trace: 是否包含执行轨迹
            
        Returns:
            根据stream_type参数返回不同类型的结果:
            - None: 返回字符串结果或带轨迹的字典
            - 'step': 返回步骤级流式生成器
            - 'token': 返回令牌级流式生成器
        """
        # 根据参数选择合适的处理方法
        if not stream_type:
            # 非流式处理
            if mode == "react":
                if include_trace:
                    return await self.process_query_with_trace(query)
                else:
                    return await self.process_query_with_react(query)
            else:
                return await self.process_query(query)
        elif stream_type == "step":
            # 步骤级流式处理
            return self.stream_process_query(query)
        elif stream_type == "token":
            # 令牌级流式处理
            return self.stream_process_query_token(query)
        else:
            raise ValueError(f"不支持的流式类型: {stream_type}")

    async def process_query(self, query: str) -> Any:
        """使用标准方法处理用户查询"""
        if not self.llm_client: 
            return "Error: Language model client not configured."

        messages = [
            {"role": "system", "content": "You are an intelligent assistant that can utilize available tools to answer questions."},
            {"role": "user", "content": query}
        ]
        
        # 获取配置的模型名称
        llm_config = self.config.get("llm_config")
        if not llm_config or not llm_config.model:
            return "Error: Language model name not configured."
            
        model_name = llm_config.model
        available_tools = await self.adapter.get_all_tools()
        provider = llm_config.provider
        logger.debug(f"Sending query to LLM ({provider}/{model_name}). Query: '{query[:50]}...'. Tools: {len(available_tools)}")

        try:
            # 确保关键字参数匹配SDK的期望
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=available_tools if available_tools else None
            )
            choice = response.choices[0]
            message = choice.message

            # 处理工具调用
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                logger.info(f"LLM requested tool call: '{function_name}'")
                
                try: 
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e: 
                    logger.error(f"Error parsing tool arguments: {e}")
                    return f"Error: Unable to parse parameters for tool '{function_name}'."
                
                try:
                    # 使用adapter的call_tool方法
                    result = await self.call_tool(function_name, function_args)
                    return str(result)
                except Exception as e: 
                    logger.error(f"Error calling tool: {e}", exc_info=True)
                    return f"Error: An internal error occurred while calling tool '{function_name}'."
            # 处理直接响应
            else: 
                logger.info("LLM provided direct response")
                return message.content.strip() if message.content else ""

        except TypeError as e:
             # 捕获之前见过的特定TypeError
             logger.error(f"TypeError during LLM call: {e}. Check SDK arguments for model '{model_name}'.", exc_info=True)
             return f"Error: Type error during language model call, please check SDK parameters. ({e})"
        except Exception as e:
            logger.error(f"Error during LLM interaction or tool processing: {e}", exc_info=True)
            return f"Error: An unexpected error occurred while processing your request. ({type(e).__name__}: {e})"

    async def process_query_with_react(self, query: str) -> str:
        """使用ReAct代理处理用户查询，支持多轮工具调用"""
        if not self.react_agent:
            logger.warning("ReAct Agent not initialized. Falling back to standard query processing.")
            return await self.process_query(query)
        
        logger.info(f"Processing query with ReAct agent: '{query[:50]}...'")
        result, trace = await self.react_agent.process_query(query)
        
        # 可选地处理跟踪以进行调试
        if trace and self.config.get("react_enable_trace", False):
            trace_str = self.react_agent._format_execution_trace(trace)
            logger.debug(f"ReAct execution trace:\n{trace_str}")
            
        return result

    async def process_query_with_trace(self, query: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """使用ReAct代理处理用户查询并返回执行跟踪"""
        if not self.react_agent:
            logger.warning("ReAct Agent not initialized. Falling back to standard query processing.")
            result = await self.process_query(query)
            return result, None
        
        logger.info(f"Processing query with ReAct agent (with trace): '{query[:50]}...'")
        # 临时启用跟踪记录
        original_trace_setting = self.react_agent.enable_trace
        self.react_agent.enable_trace = True
        
        # 处理查询
        result, trace = await self.react_agent.process_query(query)
        
        # 恢复原始设置
        self.react_agent.enable_trace = original_trace_setting
        
        return result, trace

    async def stream_process_query(self, query: str):
        """
        使用ReAct代理的流式功能处理用户查询
        
        Args:
            query: 用户查询字符串
            
        Returns:
            流式响应生成器，在每个步骤完成后立即返回结果
        """
        if not self.react_agent:
            logger.warning("ReAct Agent not initialized. Cannot execute streaming query.")
            yield {
                "thinking_step": None,
                "is_final": True,
                "result": "Error: ReAct Agent not initialized, cannot execute streaming thought process."
            }
            return
        
        logger.info(f"Starting streaming query processing with ReAct agent: '{query[:50]}...'")
        
        # 使用流式处理方法
        async for response in self.react_agent.stream_process_query(query):
            yield response
        
        logger.info(f"Streaming query processing complete: '{query[:50]}...'")

    async def stream_process_query_token(self, query: str):
        """
        使用ReAct代理的令牌流式功能处理用户查询
        
        Args:
            query: 用户查询字符串
            
        Returns:
            流式响应生成器，在生成每个令牌后立即返回结果
        """
        if not self.react_agent:
            logger.warning("ReAct Agent not initialized. Cannot execute token streaming.")
            yield {
                "token_chunk": None,
                "is_final": True,
                "result": "Error: ReAct Agent not initialized, cannot execute token streaming."
            }
            return
        
        logger.info(f"Starting token streaming with ReAct agent: '{query[:50]}...'")
        
        # 使用令牌流式处理方法
        async for response in self.react_agent.stream_process_query_token(query):
            yield response
        
        logger.info(f"Token streaming complete: '{query[:50]}...'")

    async def cleanup(self):
        """清理资源，包括停止监控任务"""
        await self.stop_monitoring()
        # 断开adapter的连接
        await self.adapter.__aexit__(None, None, None)
        logger.info("Client resources cleaned up.")
