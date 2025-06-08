import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Set, TypeVar, Generic, Protocol

logger = logging.getLogger(__name__)

# 定义一个协议，表示任何具有call_tool方法的会话类型
class SessionProtocol(Protocol):
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        ...

# 会话类型变量
SessionType = TypeVar('SessionType')

class ServiceRegistry:
    """Manages the state of connected services and their tools."""
    def __init__(self):
        self.sessions: Dict[str, Any] = {}  # server_url -> session
        self.service_health: Dict[str, datetime] = {} # server_url -> last_heartbeat_time
        self.tool_cache: Dict[str, Dict[str, Any]] = {} # tool_name -> tool_definition
        self.tool_to_session_map: Dict[str, Any] = {} # tool_name -> session
        self.service_names: Dict[str, str] = {}  # server_url -> service_name
        logger.info("ServiceRegistry initialized.")

    def add_service(self, url: str, session: Any, tools: List[Tuple[str, Dict[str, Any]]], name: str = "") -> List[str]:
        """Adds a new service, its session, and tools to the registry. Returns added tool names."""
        print(f"[DEBUG][add_service] url={url}, id(session)={id(session)}")
        if url in self.sessions:
            logger.warning(f"Attempting to add already registered service: {url}. Removing old service before overwriting.")
            self.remove_service(url)

        self.sessions[url] = session
        self.service_health[url] = datetime.now() # Mark healthy on add
        
        # Store service name
        display_name = name or url
        self.service_names[url] = display_name

        added_tool_names = []
        for tool_name, tool_definition in tools:
             if tool_name in self.tool_cache:
                 logger.warning(f"Tool name conflict: '{tool_name}' from {display_name} ({url}) conflicts with existing tool. Skipping this tool.")
                 continue
             self.tool_cache[tool_name] = tool_definition
             self.tool_to_session_map[tool_name] = session
             added_tool_names.append(tool_name)
        logger.info(f"Service '{display_name}' ({url}) added with tools: {added_tool_names}")
        return added_tool_names

    def remove_service(self, url: str) -> Optional[Any]:
        """Removes a service and its associated tools from the registry."""
        session = self.sessions.pop(url, None) # Use pop with default None
        display_name = self.service_names.get(url, url)
        
        if not session:
            logger.warning(f"Attempted to remove non-existent service: {display_name} ({url})")
            return None

        # Remove health record and service name
        if url in self.service_health:
            del self.service_health[url]
        if url in self.service_names:
            del self.service_names[url]

        # Remove associated tools efficiently
        tools_to_remove = [name for name, owner_session in self.tool_to_session_map.items() if owner_session == session]
        if tools_to_remove:
            logger.info(f"Removing tools from registry associated with {display_name} ({url}): {tools_to_remove}")
            for tool_name in tools_to_remove:
                # Check existence before deleting, although keys should be consistent
                if tool_name in self.tool_cache: del self.tool_cache[tool_name]
                if tool_name in self.tool_to_session_map: del self.tool_to_session_map[tool_name]

        logger.info(f"Service '{display_name}' ({url}) removed from registry.")
        return session

    def get_session(self, url: str) -> Optional[Any]:
        return self.sessions.get(url)
        
    def get_service_name(self, url: str) -> str:
        """Get the display name of a service"""
        return self.service_names.get(url, url)

    def get_session_for_tool(self, tool_name: str) -> Optional[Any]:
        return self.tool_to_session_map.get(tool_name)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的定义
        
        Returns:
            所有已注册工具的列表
        """
        all_tools = []
        
        # 遍历所有工具并添加服务信息
        for tool_name, tool_def in self.tool_cache.items():
            # 获取工具所属的服务信息
            session = self.tool_to_session_map.get(tool_name)
            service_url = None
            service_name = None
            
            # 查找工具所属的服务
            for url, sess in self.sessions.items():
                if sess is session:
                    service_url = url
                    service_name = self.get_service_name(url)
                    break
            
            # 创建包含服务信息的工具定义
            tool_with_service = tool_def.copy()
            
            # 如果是第一层级不包含function的情况，需要调整结构
            if "function" not in tool_with_service and isinstance(tool_with_service, dict):
                # 确保工具有一个function字段
                tool_with_service = {
                    "type": "function",
                    "function": tool_with_service
                }
            
            # 添加服务信息到函数名称或描述
            if "function" in tool_with_service:
                function_data = tool_with_service["function"]
                
                # 添加服务信息到描述中
                if service_name and service_url:
                    original_description = function_data.get("description", "")
                    if not original_description.endswith(f" (来自服务: {service_name})"):
                        function_data["description"] = f"{original_description} (来自服务: {service_name})"
                
                # 在内部保存服务信息，便于后续使用
                function_data["service_info"] = {
                    "service_url": service_url,
                    "service_name": service_name
                }
            
            all_tools.append(tool_with_service)
        
        logger.info(f"Returning {len(all_tools)} tools from {len(self.get_all_service_urls())} services")
        return all_tools
        
    def get_all_tool_info(self) -> List[Dict[str, Any]]:
        """获取所有工具的详细信息"""
        tools_info = []
        for tool_name in self.tool_cache.keys():
            # 获取工具所属的服务
            session = self.tool_to_session_map.get(tool_name)
            print(f"[DEBUG][get_all_tool_info] tool_name={tool_name}, id(session)={id(session) if session else None}")
            service_url = None
            service_name = None
            
            # 查找工具所属的服务URL和名称
            for url, sess in self.sessions.items():
                print(f"[DEBUG][get_all_tool_info]   url={url}, id(sess)={id(sess)}")
                if sess is session:
                    service_url = url
                    service_name = self.get_service_name(url)
                    break
            
            # 获取详细工具信息
            detailed_tool = self._get_detailed_tool_info(tool_name)
            if detailed_tool:
                # 添加服务信息
                detailed_tool["service_url"] = service_url
                detailed_tool["service_name"] = service_name
                tools_info.append(detailed_tool)
            
        return tools_info
        
    def get_connected_services(self) -> List[Dict[str, Any]]:
        """获取所有已连接服务的信息"""
        services = []
        for url in self.get_all_service_urls():
            tools = self.get_tools_for_service(url)
            services.append({
                "url": url,
                "name": self.get_service_name(url),
                "tool_count": len(tools)
            })
        return services

    def get_tools_for_service(self, url: str) -> List[str]:
        """Get list of tools provided by the specified service"""
        session = self.sessions.get(url)
        display_name = self.service_names.get(url, url)
        logger.info(f"Getting tools for service: {display_name} ({url})")
        print(f"[DEBUG][get_tools_for_service] url={url}, id(session)={id(session) if session else None}")
        
        if not session:
            return []
        # Find all tool names belonging to this session
        tools = [name for name, s in self.tool_to_session_map.items() if s is session]
        return tools

    def _extract_description_from_schema(self, prop_info):
        """从 schema 中提取描述信息"""
        if isinstance(prop_info, dict):
            # 优先查找 description 字段
            if 'description' in prop_info:
                return prop_info['description']
            # 其次查找 title 字段
            elif 'title' in prop_info:
                return prop_info['title']
            # 检查是否有 anyOf 或 allOf 结构
            elif 'anyOf' in prop_info:
                for item in prop_info['anyOf']:
                    if isinstance(item, dict) and 'description' in item:
                        return item['description']
            elif 'allOf' in prop_info:
                for item in prop_info['allOf']:
                    if isinstance(item, dict) and 'description' in item:
                        return item['description']

        return "无描述"

    def _extract_type_from_schema(self, prop_info):
        """从 schema 中提取类型信息"""
        if isinstance(prop_info, dict):
            if 'type' in prop_info:
                return prop_info['type']
            elif 'anyOf' in prop_info:
                # 处理 Union 类型
                types = []
                for item in prop_info['anyOf']:
                    if isinstance(item, dict) and 'type' in item:
                        types.append(item['type'])
                return '|'.join(types) if types else '未知'
            elif 'allOf' in prop_info:
                # 处理 intersection 类型
                for item in prop_info['allOf']:
                    if isinstance(item, dict) and 'type' in item:
                        return item['type']

        return "未知"

    def _get_detailed_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """获取工具的详细信息，包括参数描述、类型等"""
        tool_def = self.tool_cache.get(tool_name)
        if not tool_def:
            return {}
            
        # 基本信息
        tool_info = {
            "name": tool_name,
            "description": tool_def.get("function", {}).get("description", "无描述"),
            "parameters": []
        }
        
        # 提取参数信息
        parameters = tool_def.get("function", {}).get("parameters", {})
        if isinstance(parameters, dict):
            properties = parameters.get("properties", {})
            required_fields = parameters.get("required", [])
            
            for prop_name, prop_info in properties.items():
                param_type = self._extract_type_from_schema(prop_info)
                param_desc = self._extract_description_from_schema(prop_info)
                required = prop_name in required_fields
                
                # 提取默认值
                default_value = None
                if isinstance(prop_info, dict) and "default" in prop_info:
                    default_value = prop_info["default"]
                
                # 提取约束条件
                constraints = {}
                if isinstance(prop_info, dict):
                    for constraint in ["minimum", "maximum", "minLength", "maxLength", "pattern", "enum", "format", "ge", "le"]:
                        if constraint in prop_info:
                            constraints[constraint] = prop_info[constraint]
                
                param_info = {
                    "name": prop_name,
                    "type": param_type,
                    "description": param_desc,
                    "required": required
                }
                
                if default_value is not None:
                    param_info["default"] = default_value
                    
                if constraints:
                    param_info["constraints"] = constraints
                
                tool_info["parameters"].append(param_info)
        
        return tool_info

    def get_service_details(self, url: str) -> Dict[str, Any]:
        """Get detailed information for the specified service"""
        if url not in self.sessions:
            return {}
            
        display_name = self.service_names.get(url, url)
        logger.info(f"Getting service details for: {display_name} ({url})")
        session = self.sessions.get(url)
        print(f"[DEBUG][get_service_details] url={url}, id(session)={id(session) if session else None}")
        tools = self.get_tools_for_service(url)
        last_heartbeat = self.service_health.get(url)
        
        # 获取详细工具信息
        detailed_tools = []
        for tool_name in tools:
            detailed_tool = self._get_detailed_tool_info(tool_name)
            if detailed_tool:
                detailed_tools.append(detailed_tool)
        
        return {
            "url": url,
            "name": display_name,
            "tools": detailed_tools,
            "tool_count": len(tools),
            "last_heartbeat": str(last_heartbeat) if last_heartbeat else "N/A",
            "connected": url in self.sessions
        }

    def get_all_service_urls(self) -> List[str]:
        # Get URLs only for currently active sessions
        return list(self.sessions.keys())

    def update_service_health(self, url: str):
        """Updates the last heartbeat time for a service."""
        if url in self.sessions: # Only update health for active sessions
            self.service_health[url] = datetime.now()
            logger.debug(f"Health updated for service: {self.get_service_name(url)} ({url})")

    def get_last_heartbeat(self, url: str) -> Optional[datetime]:
        return self.service_health.get(url)

    def get_registered_services_details(self) -> List[Dict[str, Any]]:
         """Returns details for the /health endpoint."""
         details = []
         # Iterate through active sessions
         for url in self.get_all_service_urls():
             last_heartbeat = self.service_health.get(url)
             tools = self.get_tools_for_service(url)  # Get tool list for this service
             details.append({
                 "url": url,
                 "name": self.get_service_name(url),
                 "last_heartbeat": str(last_heartbeat) if last_heartbeat else "N/A",
                 "tools": tools  # Add tool list to return data
             })
         return details

    def get_tool_count(self) -> int:
         return len(self.tool_cache)

    def get_session_count(self) -> int:
         return len(self.sessions)
