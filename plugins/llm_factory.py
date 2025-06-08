import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional, Any
import logging
from config.config import LLMConfig

logger = logging.getLogger(__name__)

def create_llm_client(config: LLMConfig) -> Optional[Any]:
    """Create LLM client instance based on provider configuration"""
    if not config.api_key:
        logger.warning(f"Missing {config.provider} API key, cannot initialize LLM client")
        return None
        
    try:
        logger.info(f"正在创建LLM客户端，provider={config.provider}, model={config.model}")
        
        if config.provider == "zhipuai":
            logger.info(f"使用智谱AI API，模型={config.model}")
            try:
                from zhipuai import ZhipuAI
                client = ZhipuAI(api_key=config.api_key)
                # 验证客户端是否可用
                logger.info("正在验证智谱AI客户端...")
                try:
                    # 尝试检查客户端的基本功能
                    if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                        logger.info("智谱AI客户端验证成功")
                    else:
                        logger.warning("智谱AI客户端缺少必要的方法，可能无法正常工作")
                except Exception as ve:
                    logger.warning(f"智谱AI客户端验证时出错: {ve}")
                return client
            except Exception as e:
                logger.error(f"创建智谱AI客户端失败: {e}", exc_info=True)
                raise
            
        elif config.provider == "deepseek":
            logger.info(f"使用DeepSeek API，模型={config.model}, base_url={config.base_url or 'https://api.deepseek.com/v1'}")
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=config.api_key,
                    base_url=config.base_url or "https://api.deepseek.com/v1"
                )
                # 验证客户端是否可用
                logger.info("正在验证DeepSeek客户端...")
                try:
                    if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                        logger.info("DeepSeek客户端验证成功")
                    else:
                        logger.warning("DeepSeek客户端缺少必要的方法，可能无法正常工作")
                except Exception as ve:
                    logger.warning(f"DeepSeek客户端验证时出错: {ve}")
                return client
            except Exception as e:
                logger.error(f"创建DeepSeek客户端失败: {e}", exc_info=True)
                raise
            
        elif config.provider == "openai_compatible":
            if not config.base_url:
                logger.error("base_url is required for openai_compatible provider")
                return None
                
            logger.info(f"使用OpenAI兼容API，模型={config.model}, base_url={config.base_url}")
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=config.api_key,
                    base_url=config.base_url
                )
                # 验证客户端是否可用
                logger.info("正在验证OpenAI兼容客户端...")
                try:
                    if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                        logger.info("OpenAI兼容客户端验证成功")
                    else:
                        logger.warning("OpenAI兼容客户端缺少必要的方法，可能无法正常工作")
                except Exception as ve:
                    logger.warning(f"OpenAI兼容客户端验证时出错: {ve}")
                return client
            except Exception as e:
                logger.error(f"创建OpenAI兼容客户端失败: {e}", exc_info=True)
                raise
            
        else:
            logger.error(f"不支持的LLM提供商: {config.provider}")
            return None
            
    except ImportError as e:
        logger.error(f"无法导入{config.provider}所需模块: {e}")
        if config.provider == "zhipuai":
            logger.error("请安装智谱AI客户端: pip install zhipuai")
        elif config.provider in ["deepseek", "openai_compatible"]:
            logger.error("请安装OpenAI客户端: pip install openai")
        return None
        
    except Exception as e:
        logger.error(f"初始化{config.provider}客户端时出错: {e}", exc_info=True)
        return None 
