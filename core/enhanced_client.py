import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Enhanced FastMCP Client with ReAct support

This module provides an enhanced client that wraps FastMCP's Client and adds
ReAct (Reasoning + Acting) capabilities for improved tool usage.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Union, Tuple
from datetime import datetime, timedelta
from fastmcp import Client

from plugins.llm_factory import create_llm_client
from plugins.react_agent import ReActAgent

logger = logging.getLogger(__name__)

class EnhancedFastMCPClient:
    """
    Enhanced FastMCP client with ReAct support
    
    This class wraps FastMCP's Client and adds ReAct capabilities,
    including query processing, streaming, and health monitoring.
    """
    
    def __init__(self, config_or_url: Any, **kwargs):
        """
        Initialize the enhanced client
        
        Args:
            config_or_url: FastMCP configuration or server URL
            **kwargs: Additional configuration parameters
        """
        # Create FastMCP client
        self.client = Client(config_or_url)
        
        # Store configuration
        self.config = kwargs.get("config", {})
        self.base_url = str(config_or_url) if isinstance(config_or_url, str) else None
        
        # Initialize LLM and ReAct components
        self.llm_client = None
        self.react_agent = None
        
        # Timing configuration
        self.heartbeat_interval = timedelta(seconds=int(kwargs.get("heartbeat_interval", 60)))
        self.heartbeat_timeout = timedelta(seconds=int(kwargs.get("heartbeat_timeout", 180)))
        self.reconnection_interval = timedelta(seconds=int(kwargs.get("reconnection_interval", 60)))
        self.http_timeout = int(kwargs.get("http_timeout", 10))
        
        # Monitoring tasks
        self.heartbeat_task = None
        self.reconnection_task = None
        self.pending_reconnection = set()
        
        # Initialize LLM client if configured
        llm_config = self.config.get("llm_config")
        if llm_config:
            self._initialize_llm_client(llm_config)
    
    def _initialize_llm_client(self, llm_config):
        """Initialize LLM client and ReAct agent"""
        try:
            # Check configuration completeness
            if not llm_config.api_key or not llm_config.model:
                missing = []
                if not llm_config.api_key: missing.append("api_key")
                if not llm_config.model: missing.append("model")
                logger.error(f"LLM configuration incomplete, missing required parameters: {', '.join(missing)}")
                logger.error(f"Current config: provider={llm_config.provider}, model={llm_config.model}, base_url={llm_config.base_url}")
                logger.error("ReAct Agent will not be initialized, streaming functionality unavailable")
                return
            elif not llm_config.provider:
                logger.error("LLM configuration missing provider field")
                logger.error("ReAct Agent will not be initialized, streaming functionality unavailable")
                return
            
            self.llm_client = create_llm_client(llm_config)
            if self.llm_client:
                logger.info(f"{llm_config.provider.capitalize()} Client initialized with model {llm_config.model}.")
                # Initialize ReAct agent if LLM client is available
                try:
                    self.react_agent = ReActAgent(self.llm_client, self, self.config)
                    logger.info("ReAct Agent initialized successfully.")
                except Exception as react_err:
                    logger.error(f"ReAct Agent initialization failed: {react_err}", exc_info=True)
                    self.react_agent = None
            else:
                logger.warning(f"LLM Client initialization failed, provider={llm_config.provider}, model={llm_config.model}")
        except Exception as e:
            logger.error(f"Error creating LLM client: {e}", exc_info=True)
    
    # Async context manager support
    async def __aenter__(self):
        """
        Async context manager entry
        
        Returns:
            EnhancedFastMCPClient instance
        """
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit
        """
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    # Core FastMCP Client methods
    
    async def list_tools(self) -> List[Any]:
        """
        List available tools
        
        Returns:
            List of available tools
        """
        return await self.client.list_tools()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool with arguments
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        return await self.client.call_tool(tool_name, arguments)
    
    async def ping(self) -> bool:
        """
        Check server connection
        
        Returns:
            True if server is reachable
        """
        try:
            # FastMCP doesn't have a direct ping method, so we'll use list_tools
            await self.client.list_tools()
            return True
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            return False
    
    async def is_service_healthy(self) -> bool:
        """
        Check if service is healthy
        
        Returns:
            True if service is healthy
        """
        return await self.ping()
    
    # Enhanced functionality
    
    async def process_query(self, query: str) -> str:
        """
        Process a query using standard method
        
        Args:
            query: User query
            
        Returns:
            Query result
        """
        if not self.llm_client:
            return "Error: Language model client not configured."

        messages = [
            {"role": "system", "content": "You are an intelligent assistant that can utilize available tools to answer questions."},
            {"role": "user", "content": query}
        ]
        
        # Get configured model name
        llm_config = self.config.get("llm_config")
        if not llm_config or not llm_config.model:
            return "Error: Language model name not configured."
            
        model_name = llm_config.model
        provider = llm_config.provider
        
        try:
            # Get available tools
            tools = await self.client.list_tools()
            
            # Format tools for LLM
            available_tools = []
            for tool in tools:
                tool_dict = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, 'description', f"Tool {tool.name}"),
                        "parameters": getattr(tool, 'parameters', {}) or {}
                    }
                }
                available_tools.append(tool_dict)
            
            logger.debug(f"Sending query to LLM ({provider}/{model_name}). Query: '{query[:50]}...'. Tools: {len(available_tools)}")
            
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=available_tools if available_tools else None
            )
            
            choice = response.choices[0]
            message = choice.message
            
            # Handle tool calls
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                logger.info(f"LLM requested tool call: '{function_name}'")
                
                try:
                    import json
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing tool arguments: {e}")
                    return f"Error: Unable to parse parameters for tool '{function_name}'."
                
                try:
                    # Call tool using FastMCP client
                    result = await self.call_tool(function_name, function_args)
                    return str(result)
                except Exception as e:
                    logger.error(f"Error calling tool: {e}", exc_info=True)
                    return f"Error: An internal error occurred while calling tool '{function_name}'."
            # Handle direct responses
            else:
                logger.info("LLM provided direct response")
                return message.content.strip() if message.content else ""
                
        except Exception as e:
            logger.error(f"Error during LLM interaction or tool processing: {e}", exc_info=True)
            return f"Error: An unexpected error occurred while processing your request. ({type(e).__name__}: {e})"
    
    async def process_query_with_react(self, query: str) -> str:
        """
        Process a query using ReAct mode
        
        Args:
            query: User query
            
        Returns:
            Query result
        """
        if not self.react_agent:
            logger.warning("ReAct Agent not initialized. Falling back to standard query processing.")
            return await self.process_query(query)
        
        logger.info(f"Processing query with ReAct agent: '{query[:50]}...'")
        result, _ = await self.react_agent.process_query(query)
        return result
    
    async def process_query_with_trace(self, query: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Process a query using ReAct mode and return execution trace
        
        Args:
            query: User query
            
        Returns:
            (Query result, execution trace)
        """
        if not self.react_agent:
            logger.warning("ReAct Agent not initialized. Falling back to standard query processing.")
            result = await self.process_query(query)
            return result, None
        
        logger.info(f"Processing query with ReAct agent (with trace): '{query[:50]}...'")
        # Temporarily enable trace recording
        original_trace_setting = self.react_agent.enable_trace
        self.react_agent.enable_trace = True
        
        # Process query
        result, trace = await self.react_agent.process_query(query)
        
        # Restore original setting
        self.react_agent.enable_trace = original_trace_setting
        
        return result, trace
    
    async def stream_process_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream process a query (step-level)
        
        Args:
            query: User query
            
        Returns:
            Stream of step-level responses
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
        
        # Use streaming method
        async for response in self.react_agent.stream_process_query(query):
            yield response
        
        logger.info(f"Streaming query processing complete: '{query[:50]}...'")
    
    async def stream_process_query_token(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream process a query (token-level)
        
        Args:
            query: User query
            
        Returns:
            Stream of token-level responses
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
        
        # Use token streaming method
        async for response in self.react_agent.stream_process_query_token(query):
            yield response
        
        logger.info(f"Token streaming complete: '{query[:50]}...'")
    
    async def process_unified_query(
        self, 
        query: str, 
        mode: str = "react",
        stream_type: Optional[str] = None,
        include_trace: bool = False
    ) -> Union[str, Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Unified query processing method
        
        Args:
            query: User query
            mode: Processing mode ('standard' or 'react')
            stream_type: Stream type (None, 'step', or 'token')
            include_trace: Whether to include execution trace
            
        Returns:
            Query result in the appropriate format
        """
        # Non-streaming processing
        if not stream_type:
            if mode == "react":
                if include_trace:
                    return await self.process_query_with_trace(query)
                else:
                    return await self.process_query_with_react(query)
            else:
                return await self.process_query(query)
        # Step-level streaming
        elif stream_type == "step":
            return self.stream_process_query(query)
        # Token-level streaming
        elif stream_type == "token":
            return self.stream_process_query_token(query)
        else:
            raise ValueError(f"Unsupported stream type: {stream_type}")
    
    async def start_monitoring(self) -> None:
        """Start health monitoring"""
        # Start heartbeat monitor
        if self.heartbeat_task is None or self.heartbeat_task.done():
            logger.info(f"Starting heartbeat monitor. Interval: {self.heartbeat_interval.total_seconds()}s")
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Start reconnection monitor
        if self.reconnection_task is None or self.reconnection_task.done():
            logger.info(f"Starting reconnection monitor. Interval: {self.reconnection_interval.total_seconds()}s")
            self.reconnection_task = asyncio.create_task(self._reconnection_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring"""
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
        self.heartbeat_task = None
        self.reconnection_task = None
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat monitoring loop"""
        while True:
            await asyncio.sleep(self.heartbeat_interval.total_seconds())
            await self._check_service_health()
    
    async def _check_service_health(self) -> None:
        """Check service health"""
        logger.debug("Running periodic health check...")
        try:
            await self.ping()
            logger.debug(f"Health check SUCCESS for: {self.base_url}")
        except Exception as e:
            logger.warning(f"Health check FAILED for {self.base_url}: {e}")
            self.pending_reconnection.add(self.base_url)
    
    async def _reconnection_loop(self) -> None:
        """Reconnection loop"""
        while True:
            await asyncio.sleep(self.reconnection_interval.total_seconds())
            await self._attempt_reconnections()
    
    async def _attempt_reconnections(self) -> None:
        """Attempt reconnections"""
        if not self.pending_reconnection:
            return  # No pending reconnections
        
        # Create copy to avoid modification during iteration
        urls_to_retry = list(self.pending_reconnection)
        logger.info(f"Attempting to reconnect {len(urls_to_retry)} service(s): {urls_to_retry}")
        
        for url in urls_to_retry:
            try:
                await self.ping()
                logger.info(f"Reconnection successful for: {url}")
                self.pending_reconnection.discard(url)
            except Exception as e:
                logger.warning(f"Reconnection attempt failed for {url}: {e}")
                # Keep URL in pending_reconnection for next cycle
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.stop_monitoring()
        # Close FastMCP client
        await self.client.__aexit__(None, None, None)
        logger.info("Client resources cleaned up.") 
