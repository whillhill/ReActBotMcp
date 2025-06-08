import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
ReAct (Reasoning + Acting) mode implementation for enhancing MCP client reasoning and tool calling capabilities.
"""

import json
import logging
import uuid
from typing import Dict, List, Any, Tuple, Optional, AsyncGenerator
from datetime import datetime
import asyncio

from fastmcp import Client

logger = logging.getLogger(__name__)

class ReActAgent:
    """
    Implementation of ReAct (Reasoning + Acting) mode agent.
    Enhances LLM reasoning and tool calling capabilities, supporting multi-round tool calling cycles.
    """
    
    def __init__(self, llm_client, client, config, registry=None):
        """
        Initialize ReAct agent
        
        Args:
            llm_client: LLM client instance
            client: Client instance (FastMCP Client or EnhancedFastMCPClient)
            config: Configuration parameters
            registry: ServiceRegistry instance (optional) - if provided, will get tools from global registry
        """
        self.llm_client = llm_client
        self.client = client
        self.config = config
        self.registry = registry
        
        # Read ReAct related parameters from configuration
        self.max_iterations = config.get("react_max_iterations", 25)
        self.enable_trace = config.get("react_enable_trace", False)
        
    def _create_react_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Create system prompt for ReAct mode
        
        Args:
            tools: List of available tools
            
        Returns:
            Optimized system prompt string
        """
        base_prompt = """You are an intelligent assistant, using available tools to solve problems. Follow these steps:

1. THINKING: Analyze the problem, determine which tools and methods to use
2. ACTION: Choose an appropriate tool and use it
3. OBSERVATION: Analyze the results returned by the tool
4. Repeat steps 1-3 until you can provide a complete answer
5. ANSWER: Synthesize all information to provide the final answer

If the question is simple and doesn't require tools, answer directly. If tools are needed, follow the steps above."""
        
        # Add tools list information
        if tools:
            tools_str = "\n".join([f"- {tool['function']['name']}: {tool['function']['description']}" 
                                for tool in tools if 'function' in tool])
            base_prompt += f"\n\nAvailable tools:\n{tools_str}"
        
        return base_prompt
    
    def _format_execution_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Format execution trace for debugging or reporting
        
        Args:
            trace: Execution trace list
            
        Returns:
            Formatted execution trace string
        """
        formatted = []
        for step in trace:
            if step["role"] == "assistant":
                formatted.append(f"Thinking: {step['content']}")
            elif step["role"] == "tool":
                formatted.append(f"Tool {step['name']}: {step['result']}")
        return "\n".join(formatted)
    
    def _enhance_tool_description(self, description: str, tool_name: str) -> str:
        """
        Enhance tool description to make it more suitable for ReAct mode
        
        Args:
            description: Original tool description
            tool_name: Tool name
            
        Returns:
            Enhanced tool description
        """
        if not description.endswith('.'):
            description += '.'
            
        # If description doesn't include "use this tool" guidance, add it
        if "use this tool" not in description.lower():
            description += f" Use this tool when you need {tool_name} related functionality."
            
        return description
    
    async def process_tool_definitions(self, tools) -> List[Dict[str, Any]]:
        """
        Process tool definitions, optimize descriptions for ReAct mode
        
        Args:
            tools: Tool list from client.list_tools() or registry.get_all_tools()
            
        Returns:
            Processed tool definition list for LLM
        """
        processed_tools = []
        
        for tool in tools:
            # Handle dictionary format tools (from registry.get_all_tools())
            if isinstance(tool, dict):
                # If already in LLM tool format, add directly with enhanced description
                if "type" in tool and tool.get("type") == "function" and "function" in tool:
                    # Get function description and enhance
                    function_def = tool["function"]
                    if "description" in function_def:
                        function_def["description"] = self._enhance_tool_description(
                            function_def["description"], 
                            function_def["name"]
                        )
                    processed_tools.append(tool)
                    continue
                
                # Otherwise, try to extract name, description and parameters from tool dict
                tool_name = tool.get("name") or tool.get("function", {}).get("name")
                tool_desc = tool.get("description") or tool.get("function", {}).get("description", f"Tool {tool_name}")
                parameters = tool.get("parameters") or tool.get("function", {}).get("parameters") or tool.get("inputSchema", {}) or {}
                
                if not tool_name:
                    logger.warning(f"Skipping tool with missing name: {tool}")
                    continue
                
                # Enhance description
                enhanced_desc = self._enhance_tool_description(tool_desc, tool_name)
                
                # Ensure parameters format is correct
                if not isinstance(parameters, dict) or parameters.get("type") != "object":
                    parameters = {
                        "type": "object", 
                        "properties": parameters, 
                        "required": list(parameters.keys()) if isinstance(parameters, dict) else []
                    }
                
                # Create LLM tool definition
                tool_definition = {
                    "type": "function", 
                    "function": {
                        "name": tool_name, 
                        "description": enhanced_desc, 
                        "parameters": parameters
                    }
                }
                processed_tools.append(tool_definition)
            else:
                # Handle object format tools (from client.list_tools())
                # For FastMCP tools
                tool_name = getattr(tool, 'name', None)
                if not tool_name:
                    logger.warning(f"Skipping tool with missing name: {tool}")
                    continue
                
                # Get parameters schema
                parameters = {}
                # FastMCP tools have 'parameters' attribute
                if hasattr(tool, 'parameters'):
                    parameters = getattr(tool, 'parameters', {}) or {}
                # Fallback to inputSchema for backward compatibility
                elif hasattr(tool, 'inputSchema'):
                    parameters = getattr(tool, 'inputSchema', {}) or {}
                
                # Ensure parameters format is correct
                if not isinstance(parameters, dict) or parameters.get("type") != "object":
                    parameters = {
                        "type": "object", 
                        "properties": parameters, 
                        "required": list(parameters.keys()) if isinstance(parameters, dict) else []
                    }
                
                # Get tool description
                description = getattr(tool, 'description', f"Tool {tool_name}")
                
                # Enhance description
                enhanced_desc = self._enhance_tool_description(description, tool_name)
                    
                # Create LLM tool definition
                tool_definition = {
                    "type": "function", 
                    "function": {
                        "name": tool_name, 
                        "description": enhanced_desc, 
                        "parameters": parameters
                    }
                }
                processed_tools.append(tool_definition)
            
        return processed_tools
        
    async def _call_tool_with_registry(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """
        Call tool using registry or client
        
        Args:
            function_name: Tool name
            function_args: Tool parameters
            
        Returns:
            Tool execution result
        """
        # If registry is available, use it to find the appropriate session
        if self.registry:
            session = self.registry.get_session_for_tool(function_name)
            if session:
                logger.info(f"Found tool '{function_name}' in registry, calling through appropriate session")
                try:
                    return await session.call_tool(function_name, function_args)
                except Exception as e:
                    logger.error(f"Error calling tool '{function_name}' through registry session: {e}")
                    raise e
            else:
                logger.warning(f"Tool '{function_name}' not found in registry")
                raise ValueError(f"Tool not found: {function_name}")
        
        # If no registry or tool not found, fall back to direct client call
        logger.info(f"Calling tool '{function_name}' directly through client")
        
        # Handle both FastMCP Client and EnhancedFastMCPClient
        from core.enhanced_client import EnhancedFastMCPClient
        if isinstance(self.client, (Client, EnhancedFastMCPClient)):
            result = await self.client.call_tool(function_name, function_args)
            
            # FastMCP Client returns a list of content objects
            if isinstance(result, list) and len(result) > 0:
                # Extract text content if available
                if hasattr(result[0], 'text'):
                    return result[0].text
                # Fall back to string representation
                return str(result)
            
            return result
        else:
            # Legacy client support
            return await self.client.call_tool(function_name, function_args)

    async def process_query(self, query: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Process query using ReAct mode, supporting multi-round tool calls
        
        Args:
            query: User query string
            
        Returns:
            (Result string, execution trace list if enabled)
        """
        if not self.llm_client: 
            return "Error: Language model client not configured.", None
        
        # Get available tools
        if self.registry:
            # Get all tools from global registry
            raw_tools = self.registry.get_all_tools()
            logger.info(f"Using {len(raw_tools)} tools from global registry")
        else:
            # Backward compatibility: get tools from single client
            raw_tools = await self.client.list_tools()
            logger.info(f"Using {len(raw_tools)} tools from single client")
        
        available_tools = await self.process_tool_definitions(raw_tools)
        
        # Build system prompt
        system_prompt = self._create_react_system_prompt(available_tools)
        
        # Initial message history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Record execution trace
        execution_trace = []
        
        # ReAct loop
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Starting ReAct iteration #{iterations}, processing query: '{query[:50]}...'")
            
            # Call LLM
            try:
                # Get model name
                llm_config = self.config.get("llm_config")
                if not llm_config or not llm_config.model:
                    return "Error: Language model name not configured.", None
                    
                model_name = llm_config.model
                provider = llm_config.provider
                
                logger.debug(f"Sending query to LLM ({provider}/{model_name}). Query: '{query[:50]}...'. Tools: {len(available_tools)}")
                
                # Create LLM request
                response = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=available_tools if available_tools else None
                )
                
                choice = response.choices[0]
                message = choice.message
                
                # Record assistant response
                assistant_response = message.content or ""
                if self.enable_trace:
                    execution_trace.append({"role": "assistant", "content": assistant_response})
                
                # Tool call needed
                if choice.finish_reason == "tool_calls" and message.tool_calls:
                    # Get tool call information
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    logger.info(f"LLM requested tool call: '{function_name}'")
                    
                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    })
                    
                    # Execute tool call
                    try:
                        # Parse arguments
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Call tool using registry or client
                        try:
                            logger.info(f"Executing tool '{function_name}', parameters: {function_args}")
                            result = await self._call_tool_with_registry(function_name, function_args)
                            logger.info(f"Tool '{function_name}' returned result: {result}")
                            
                            # Process tool result
                            tool_result = str(result)
                        except Exception as e:
                            tool_result = f"Error: An internal error occurred while calling tool '{function_name}': {str(e)}"
                            logger.error(f"Tool call error: {e}", exc_info=True)
                    except json.JSONDecodeError as e:
                        tool_result = f"Error: Unable to parse parameters for tool '{function_name}': {e}"
                        logger.error(f"Parameter parsing error: {e}")
                    
                    # Record tool result in trace
                    if self.enable_trace:
                        execution_trace.append({
                            "role": "tool",
                            "name": function_name,
                            "result": tool_result
                        })
                    
                    # Add tool result to message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    
                    # Continue to next iteration
                    continue
                
                # LLM provides final answer
                logger.info(f"LLM provided final answer, ending ReAct loop")
                result = message.content.strip() if message.content else ""
                return result, execution_trace if self.enable_trace else None
                
            except TypeError as e:
                # Catch the specific TypeError seen before
                error_msg = f"Type error during LLM call: {e}. Check SDK parameters for model '{model_name}'."
                logger.error(error_msg, exc_info=True)
                return f"Error: Type error during language model call, please check SDK parameters. ({e})", None
            except Exception as e:
                error_msg = f"Error during ReAct process: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                return f"Error processing your request. ({error_msg})", None
        
        # Maximum iterations reached
        logger.warning(f"Maximum ReAct iterations reached ({self.max_iterations})")
        last_message = messages[-1].get("content", "") if messages else ""
        result = f"Processing your request exceeded the maximum iteration limit ({self.max_iterations}). " + last_message
        return result, execution_trace if self.enable_trace else None 

    async def stream_process_query(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with streaming using ReAct mode, returning results immediately after each step completes
        
        Args:
            query: User query string
            
        Yields:
            Stream responses with thinking steps and final result
        """
        if not self.llm_client: 
            yield {"thinking_step": None, "is_final": True, "result": "Error: Language model client not configured."}
            return
        
        # 获取可用工具
        if self.registry:
            # 从全局注册表获取所有工具
            raw_tools = self.registry.get_all_tools()
            logger.info(f"Using {len(raw_tools)} tools from global registry")
        else:
            # 向后兼容：从单个客户端获取工具
            raw_tools = await self.client.list_tools()
            logger.info(f"Using {len(raw_tools)} tools from single client")
        
        available_tools = await self.process_tool_definitions(raw_tools)
        
        # Build system prompt
        system_prompt = self._create_react_system_prompt(available_tools)
        
        # Initial message history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # ReAct loop
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Starting streaming ReAct iteration #{iterations}, processing query: '{query[:50]}...'")
            
            # Call LLM
            try:
                # Get model name
                llm_config = self.config.get("llm_config")
                if not llm_config or not llm_config.model:
                    yield {"thinking_step": None, "is_final": True, "result": "Error: Language model name not configured."}
                    return
                    
                model_name = llm_config.model
                provider = llm_config.provider
                
                logger.debug(f"Sending streaming query to LLM ({provider}/{model_name}). Query: '{query[:50]}...'. Tools: {len(available_tools)}")
                
                # Create LLM request
                response = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=available_tools if available_tools else None
                )
                
                choice = response.choices[0]
                message = choice.message
                
                # Record assistant response
                assistant_response = message.content or ""
                
                # Generate thinking step ID
                step_id = f"step-{uuid.uuid4()}"
                
                # Send as thinking step - start
                thinking_step = {
                    "type": "thinking",
                    "content": assistant_response,
                    "id": step_id,
                    "status": "start"
                }
                yield {"thinking_step": thinking_step, "is_final": False}
                
                # Tool call needed
                if choice.finish_reason == "tool_calls" and message.tool_calls:
                    # Send as thinking step - complete
                    thinking_step["status"] = "complete"
                    yield {"thinking_step": thinking_step, "is_final": False}
                    
                    # Get tool call information
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    logger.info(f"LLM requested tool call: '{function_name}'")
                    
                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    })
                    
                    # Create tool call step ID
                    tool_step_id = f"tool-{uuid.uuid4()}"
                    
                    # Parse tool parameters
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        function_args = {"error": f"Unable to parse parameters: {e}"}
                    
                    # Send tool call step - start
                    tool_step = {
                        "type": "tool_call",
                        "tool": function_name,
                        "id": tool_step_id,
                        "status": "start",
                        "params": function_args  # Add parameters field
                    }
                    yield {"thinking_step": tool_step, "is_final": False}
                    
                    # Execute tool call
                    try:
                        # 使用注册表或客户端调用工具
                        logger.info(f"Executing tool '{function_name}', parameters: {function_args}")
                        result = await self._call_tool_with_registry(function_name, function_args)
                        logger.info(f"Tool '{function_name}' returned result: {result}")
                        
                        # 处理工具返回结果
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error: An internal error occurred while calling tool '{function_name}': {str(e)}"
                        logger.error(f"Tool call error: {e}", exc_info=True)
                    
                    # Send tool call step - complete
                    tool_step["result"] = tool_result
                    tool_step["status"] = "complete"
                    yield {"thinking_step": tool_step, "is_final": False}
                    
                    # Add tool result to message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    
                    # Continue to next iteration
                    continue
                
                # LLM provides final answer, update step status, then return result
                logger.info(f"LLM provided final answer, ending streaming ReAct loop")
                
                # Send as thinking step - complete
                thinking_step["status"] = "complete"
                yield {"thinking_step": thinking_step, "is_final": False}
                
                # Send final result
                result = message.content.strip() if message.content else ""
                yield {"thinking_step": None, "is_final": True, "result": result}
                return
                
            except TypeError as e:
                # Catch the specific TypeError seen before
                error_msg = f"Type error during LLM call: {e}. Check SDK parameters for model '{model_name}'."
                logger.error(error_msg, exc_info=True)
                yield {"thinking_step": None, "is_final": True, "result": f"Error: Type error during language model call, please check SDK parameters. ({e})"}
                return
            except Exception as e:
                error_msg = f"Error during streaming ReAct process: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                yield {"thinking_step": None, "is_final": True, "result": f"Error processing your request. ({error_msg})"}
                return
        
        # Maximum iterations reached
        logger.warning(f"Maximum ReAct iterations reached ({self.max_iterations})")
        last_message = messages[-1].get("content", "") if messages else ""
        result = f"Processing your request exceeded the maximum iteration limit ({self.max_iterations}). " + last_message
        yield {"thinking_step": None, "is_final": True, "result": result} 

    async def stream_process_query_token(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process query with token streaming using ReAct mode, returning tokens immediately
        
        Args:
            query: User query string
            
        Yields:
            Stream responses with token chunks
        """
        if not self.llm_client: 
            yield {"token_chunk": None, "is_final": True, "result": "Error: Language model client not configured."}
            return
        
        # 获取可用工具
        if self.registry:
            # 从全局注册表获取所有工具
            raw_tools = self.registry.get_all_tools()
            logger.info(f"Using {len(raw_tools)} tools from global registry")
        else:
            # 向后兼容：从单个客户端获取工具
            raw_tools = await self.client.list_tools()
            logger.info(f"Using {len(raw_tools)} tools from single client")
        
        available_tools = await self.process_tool_definitions(raw_tools)
        
        # 构建系统提示
        system_prompt = self._create_react_system_prompt(available_tools)
        system_prompt += "\n\nWhen thinking, surround your thoughts with <think></think> tags."
        
        # 初始消息历史
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # ReAct循环
        iterations = 0
        final_answer = ""
        thinking_buffer = ""
        current_thinking_id = None
        in_thinking_mode = False
        
        async def sync_to_async_iter(sync_iter):
            loop = asyncio.get_event_loop()
            it = iter(sync_iter)
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, it)
                except StopIteration:
                    break
                yield chunk
        
        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Starting token streaming ReAct iteration #{iterations}, query: '{query[:50]}...'")
            
            try:
                # 获取模型名
                llm_config = self.config.get("llm_config")
                if not llm_config or not llm_config.model:
                    yield {"token_chunk": None, "is_final": True, "result": "Error: Language model name not configured."}
                    return
                    
                model_name = llm_config.model
                
                # 创建流式LLM请求
                stream = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    stream=True  # 启用流式输出
                )
                
                content_buffer = ""
                current_message = {"role": "assistant", "content": ""}
                tool_calls = []
                is_function_calling = False
                
                # 判断stream是否为异步可迭代对象
                if hasattr(stream, "__aiter__"):
                    chunk_iter = stream
                    is_async = True
                else:
                    chunk_iter = sync_to_async_iter(stream)
                    is_async = False
                
                # 用统一的异步for循环处理token
                async for chunk in chunk_iter:
                    # 处理tokens
                    delta = chunk.choices[0].delta
                    
                    # 检查是否有新的内容
                    if hasattr(delta, "content") and delta.content is not None:
                        token = delta.content
                        if token is None:
                            logger.warning("Received None token from LLM stream, skipping.")
                            continue
                        content_buffer += token or ""
                        current_message["content"] += token or ""
                        
                        # 检查思考模式标记
                        if "<think>" in (token or "") and not in_thinking_mode:
                            in_thinking_mode = True
                            current_thinking_id = f"think-{uuid.uuid4()}"
                            # 发送思考开始标记
                            yield {
                                "thinking_step": {
                                    "type": "thinking",
                                    "id": current_thinking_id,
                                    "content": "",
                                    "status": "start"
                                },
                                "is_final": False
                            }
                            thinking_buffer = ""
                            continue
                        
                        if "</think>" in (token or "") and in_thinking_mode:
                            in_thinking_mode = False
                            # 发送思考结束标记
                            yield {
                                "thinking_step": {
                                    "type": "thinking",
                                    "id": current_thinking_id,
                                    "content": thinking_buffer,
                                    "status": "complete"
                                },
                                "is_final": False
                            }
                            current_thinking_id = None
                            continue
                        
                        # 记录思考内容
                        if in_thinking_mode:
                            clean_token = (token or "").replace("<think>", "").replace("</think>", "")
                            thinking_buffer += clean_token or ""
                            # 发送思考内容token
                            yield {
                                "token_chunk": {
                                    "type": "thinking",
                                    "content": clean_token or "",
                                    "thinking_id": current_thinking_id
                                },
                                "is_final": False
                            }
                        else:
                            # 发送普通回答token
                            clean_token = (token or "").replace("<think>", "").replace("</think>", "")
                            if clean_token:
                                final_answer += clean_token or ""
                                yield {
                                    "token_chunk": {
                                        "type": "content",
                                        "content": clean_token or ""
                                    },
                                    "is_final": False
                                }
                    
                    # 检查工具调用
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        is_function_calling = True
                        
                        for tool_call_delta in delta.tool_calls:
                            tool_call_id = tool_call_delta.index
                            
                            # 确保工具调用列表有足够空间
                            while len(tool_calls) <= tool_call_id:
                                tool_calls.append({
                                    "id": str(uuid.uuid4()),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            # 更新函数名
                            if hasattr(tool_call_delta, "function") and hasattr(tool_call_delta.function, "name"):
                                if tool_call_delta.function.name is not None:
                                    tool_calls[tool_call_id]["function"]["name"] = tool_call_delta.function.name
                                else:
                                    logger.warning(f"Received None as tool_call function.name for tool_call_id={tool_call_id}, skipping assignment.")
                            
                            # 更新参数
                            if hasattr(tool_call_delta, "function") and hasattr(tool_call_delta.function, "arguments"):
                                tool_calls[tool_call_id]["function"]["arguments"] += tool_call_delta.function.arguments or ""
                    
                    # 检查是否完成
                    if chunk.choices[0].finish_reason:
                        break
                
                # 流完成，检查是否需要工具调用
                # 过滤掉function.name为None的tool_call，避免400错误
                valid_tool_calls = [tc for tc in tool_calls if tc["function"].get("name") is not None]
                if is_function_calling and valid_tool_calls:
                    # 更新消息历史添加工具调用
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": valid_tool_calls
                    })
                    
                    # 处理所有工具调用
                    for tool_call in valid_tool_calls:
                        function_name = tool_call["function"]["name"]
                        function_args_str = tool_call["function"]["arguments"]
                        tool_call_id = tool_call["id"]
                        
                        # 发送工具调用开始标记
                        tool_step_id = f"tool-{uuid.uuid4()}"
                        
                        try:
                            # 解析参数
                            function_args = json.loads(function_args_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {e}, 参数字符串: {function_args_str}")
                            function_args = {}
                        
                        yield {
                            "thinking_step": {
                                "type": "tool_call",
                                "tool": function_name,
                                "id": tool_step_id,
                                "params": function_args,
                                "status": "start"
                            },
                            "is_final": False
                        }
                        
                        # 执行工具调用
                        try:
                            # 使用注册表或客户端调用工具
                            logger.info(f"Executing tool '{function_name}', parameters: {function_args}")
                            result = await self._call_tool_with_registry(function_name, function_args)
                            logger.info(f"Tool '{function_name}' returned result: {result}")
                            
                            # 处理工具返回结果
                            tool_result = str(result)
                        except Exception as e:
                            tool_result = f"Error: An internal error occurred while calling tool '{function_name}': {str(e)}"
                            logger.error(f"Tool call error: {e}", exc_info=True)
                        
                        # 发送工具调用完成标记
                        yield {
                            "thinking_step": {
                                "type": "tool_call",
                                "tool": function_name,
                                "id": tool_step_id,
                                "params": function_args,
                                "result": tool_result,
                                "status": "complete"
                            },
                            "is_final": False
                        }
                        
                        # 添加工具结果到消息历史
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_result
                        })
                    
                    # 继续下一次迭代
                    continue
                else:
                    # 没有工具调用，更新历史记录
                    messages.append(current_message)
                    
                    # 发送最终结果
                    yield {
                        "token_chunk": None,
                        "is_final": True,
                        "result": final_answer
                    }
                    return
                
            except Exception as e:
                error_msg = f"Error during token streaming: {type(e).__name__}: {e}"
                logger.error(error_msg, exc_info=True)
                yield {
                    "token_chunk": None,
                    "is_final": True,
                    "result": f"Error processing your request. ({error_msg})"
                }
                return
        
        # 达到最大迭代次数
        logger.warning(f"Maximum ReAct iterations reached ({self.max_iterations})")
        yield {
            "token_chunk": None,
            "is_final": True,
            "result": final_answer or f"Processing exceeded maximum iteration limit ({self.max_iterations})."
        } 
