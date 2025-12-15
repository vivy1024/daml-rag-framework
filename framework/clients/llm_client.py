# -*- coding: utf-8 -*-
"""
LLM调用模块

支持多个LLM提供商:
- DeepSeek (teacher模型，经济实惠)
- Ollama (student模型，本地部署)
- Moonshot Kimi (备用，长文本)

作者: BUILD_BODY Team
版本: v1.0.0
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, AsyncIterator
import httpx

logger = logging.getLogger(__name__)


class LLMConfig:
    """LLM配置 - 从环境变量读取"""

    # DeepSeek (teacher模型)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Ollama (student模型)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    # Moonshot (备用)
    MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
    MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
    MOONSHOT_MODEL = os.getenv("MOONSHOT_MODEL", "moonshot-v1-32k")

    # 通义千问 (备用)
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")

    # 通用配置
    TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60.0"))
    MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    @classmethod
    def validate(cls):
        """验证必需的API密钥是否已配置"""
        if not cls.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY未配置，DeepSeek功能将不可用")
        if not cls.MOONSHOT_API_KEY:
            logger.warning("MOONSHOT_API_KEY未配置，Moonshot功能将不可用")
        if not cls.QWEN_API_KEY:
            logger.warning("QWEN_API_KEY未配置，通义千问功能将不可用")
    
    @classmethod
    def validate_dependencies(cls):
        """
        验证所有必需的依赖项是否存在
        
        Raises:
            ImportError: 如果缺少必需的依赖模块
        """
        required_modules = ["httpx", "json", "asyncio", "logging"]
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
                logger.error(f"缺少必需模块: {module}")
        
        if missing_modules:
            raise ImportError(
                f"缺少必需的依赖模块: {', '.join(missing_modules)}。"
                f"请运行: pip install {' '.join(missing_modules)}"
            )


async def call_deepseek(
    query: str,
    few_shot_examples: List[Dict[str, Any]],
    tool_results: Dict[str, Any],
    system_prompt: str = "你是一位专业的健身教练，擅长根据用户档案提供个性化的训练建议。",
    max_tokens: int = LLMConfig.MAX_TOKENS
) -> str:
    """
    调用DeepSeek API (teacher模型)

    Args:
        query: 用户查询
        few_shot_examples: Few-Shot示例列表
        tool_results: 工具调用结果
        system_prompt: 系统提示词
        max_tokens: 最大生成token数

    Returns:
        str: AI回答
    """
    # 检查API密钥
    if not LLMConfig.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY未配置，请在.env文件中配置")

    try:
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 添加Few-Shot示例
        for example in few_shot_examples:
            messages.append({
                "role": "user",
                "content": example.get("query", "")
            })
            messages.append({
                "role": "assistant",
                "content": example.get("response", "")
            })

        # 添加工具结果到上下文
        tool_context = _format_tool_results(tool_results)
        if tool_context:
            messages.append({
                "role": "system",
                "content": f"工具调用结果:\n{tool_context}"
            })

        # 添加当前查询
        messages.append({
            "role": "user",
            "content": query
        })

        # 调用DeepSeek API
        async with httpx.AsyncClient(timeout=LLMConfig.TIMEOUT) as client:
            response = await client.post(
                f"{LLMConfig.DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLMConfig.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLMConfig.DEEPSEEK_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": LLMConfig.TEMPERATURE
                }
            )

            response.raise_for_status()
            result = response.json()

            answer = result["choices"][0]["message"]["content"]

            logger.info(
                f"DeepSeek调用成功: "
                f"tokens={result.get('usage', {}).get('total_tokens', 0)}, "
                f"length={len(answer)}"
            )

            return answer

    except httpx.HTTPError as e:
        # 记录详细的错误上下文
        logger.error(
            f"DeepSeek HTTP错误: {e}\n"
            f"上下文信息:\n"
            f"  - 模型: {LLMConfig.DEEPSEEK_MODEL}\n"
            f"  - 查询长度: {len(query)}\n"
            f"  - Few-Shot示例数: {len(few_shot_examples)}\n"
            f"  - 工具结果数: {len(tool_results) if tool_results else 0}\n"
            f"  - Max Tokens: {max_tokens}\n"
            f"  - Temperature: {LLMConfig.TEMPERATURE}",
            exc_info=True
        )
        # 返回降级响应而不是抛出异常
        return get_fallback_response(
            query=query,
            tool_results=tool_results,
            reason=f"DeepSeek HTTP错误: {str(e)}"
        )
    except Exception as e:
        # 记录详细的错误上下文
        logger.error(
            f"DeepSeek调用失败: {e}\n"
            f"上下文信息:\n"
            f"  - 模型: {LLMConfig.DEEPSEEK_MODEL}\n"
            f"  - 查询: {query[:100]}...\n"
            f"  - Few-Shot示例数: {len(few_shot_examples)}\n"
            f"  - 系统提示词长度: {len(system_prompt)}",
            exc_info=True
        )
        # 返回降级响应而不是抛出异常
        return get_fallback_response(
            query=query,
            tool_results=tool_results,
            reason=f"DeepSeek调用失败: {str(e)}"
        )


async def call_ollama(
    query: str,
    few_shot_examples: List[Dict[str, Any]],
    tool_results: Dict[str, Any],
    system_prompt: str = "你是一位专业的健身教练，擅长根据用户档案提供个性化的训练建议。",
    model: str = LLMConfig.OLLAMA_MODEL
) -> str:
    """
    调用Ollama API (student模型)

    Args:
        query: 用户查询
        few_shot_examples: Few-Shot示例列表
        tool_results: 工具调用结果
        system_prompt: 系统提示词
        model: Ollama模型名称

    Returns:
        str: AI回答
    """
    try:
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 添加Few-Shot示例
        for example in few_shot_examples:
            messages.append({
                "role": "user",
                "content": example.get("query", "")
            })
            messages.append({
                "role": "assistant",
                "content": example.get("response", "")
            })

        # 添加工具结果到上下文
        tool_context = _format_tool_results(tool_results)
        if tool_context:
            messages.append({
                "role": "system",
                "content": f"工具调用结果:\n{tool_context}"
            })

        # 添加当前查询
        messages.append({
            "role": "user",
            "content": query
        })

        # 调用Ollama API
        async with httpx.AsyncClient(timeout=LLMConfig.TIMEOUT) as client:
            response = await client.post(
                f"{LLMConfig.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )

            response.raise_for_status()
            result = response.json()

            answer = result["message"]["content"]

            logger.info(
                f"Ollama调用成功: model={model}, length={len(answer)}"
            )

            return answer

    except httpx.HTTPError as e:
        # 记录详细的错误上下文
        logger.error(
            f"Ollama HTTP错误: {e}\n"
            f"上下文信息:\n"
            f"  - 模型: {model}\n"
            f"  - 基础URL: {LLMConfig.OLLAMA_BASE_URL}\n"
            f"  - 查询长度: {len(query)}\n"
            f"  - Few-Shot示例数: {len(few_shot_examples)}\n"
            f"  - 工具结果数: {len(tool_results) if tool_results else 0}",
            exc_info=True
        )
        # 返回降级响应而不是抛出异常
        return get_fallback_response(
            query=query,
            tool_results=tool_results,
            reason=f"Ollama HTTP错误: {str(e)}"
        )
    except Exception as e:
        # 记录详细的错误上下文
        logger.error(
            f"Ollama调用失败: {e}\n"
            f"上下文信息:\n"
            f"  - 模型: {model}\n"
            f"  - 查询: {query[:100]}...\n"
            f"  - Few-Shot示例数: {len(few_shot_examples)}\n"
            f"  - 系统提示词长度: {len(system_prompt)}",
            exc_info=True
        )
        # 返回降级响应而不是抛出异常
        return get_fallback_response(
            query=query,
            tool_results=tool_results,
            reason=f"Ollama调用失败: {str(e)}"
        )


async def call_moonshot(
    query: str,
    few_shot_examples: List[Dict[str, Any]],
    tool_results: Dict[str, Any],
    system_prompt: str = "你是一位专业的健身教练。"
) -> str:
    """
    调用Moonshot API (备用，适合长文本)

    Args:
        query: 用户查询
        few_shot_examples: Few-Shot示例列表
        tool_results: 工具调用结果
        system_prompt: 系统提示词

    Returns:
        str: AI回答
    """
    # 检查API密钥
    if not LLMConfig.MOONSHOT_API_KEY:
        raise ValueError("MOONSHOT_API_KEY未配置，请在.env文件中配置")

    try:
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 添加Few-Shot示例
        for example in few_shot_examples:
            messages.append({
                "role": "user",
                "content": example.get("query", "")
            })
            messages.append({
                "role": "assistant",
                "content": example.get("response", "")
            })

        # 添加工具结果
        tool_context = _format_tool_results(tool_results)
        if tool_context:
            messages.append({
                "role": "system",
                "content": f"工具调用结果:\n{tool_context}"
            })

        messages.append({
            "role": "user",
            "content": query
        })

        async with httpx.AsyncClient(timeout=LLMConfig.TIMEOUT) as client:
            response = await client.post(
                f"{LLMConfig.MOONSHOT_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLMConfig.MOONSHOT_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLMConfig.MOONSHOT_MODEL,
                    "messages": messages,
                    "max_tokens": LLMConfig.MAX_TOKENS,
                    "temperature": LLMConfig.TEMPERATURE
                }
            )

            response.raise_for_status()
            result = response.json()

            return result["choices"][0]["message"]["content"]

    except Exception as e:
        # 记录详细的错误上下文
        logger.error(
            f"Moonshot调用失败: {e}\n"
            f"上下文信息:\n"
            f"  - 模型: {LLMConfig.MOONSHOT_MODEL}\n"
            f"  - 查询: {query[:100]}...\n"
            f"  - Few-Shot示例数: {len(few_shot_examples)}\n"
            f"  - 系统提示词长度: {len(system_prompt)}\n"
            f"  - Max Tokens: {LLMConfig.MAX_TOKENS}",
            exc_info=True
        )
        # 返回降级响应而不是抛出异常
        return get_fallback_response(
            query=query,
            tool_results=tool_results,
            reason=f"Moonshot调用失败: {str(e)}"
        )


def _format_tool_results(tool_results: Dict[str, Any]) -> str:
    """
    格式化工具结果为可读文本

    Args:
        tool_results: 工具调用结果字典

    Returns:
        str: 格式化的文本
    """
    if not tool_results:
        return ""

    lines = []
    for tool_name, result in tool_results.items():
        lines.append(f"【{tool_name}】")

        if isinstance(result, dict):
            for key, value in result.items():
                lines.append(f"  - {key}: {value}")
        elif isinstance(result, list):
            for item in result:
                lines.append(f"  - {item}")
        else:
            lines.append(f"  {result}")

    return "\n".join(lines)


def get_fallback_response(
    query: str,
    tool_results: Optional[Dict[str, Any]] = None,
    reason: str = "LLM服务暂时不可用"
) -> str:
    """
    获取降级响应（基于查询和工具结果生成有意义的默认响应）
    
    Args:
        query: 用户查询
        tool_results: 工具调用结果（可选）
        reason: 降级原因
    
    Returns:
        str: 有意义的降级响应文本
    """
    logger.warning(f"使用降级响应: {reason}")
    
    # 基础响应模板
    response_parts = [
        f"抱歉，AI分析功能暂时不可用（{reason}）。",
        "",
        f"📝 您的查询：{query}",
        ""
    ]
    
    # 如果有工具结果，提供结构化信息
    if tool_results:
        response_parts.append("✅ 已完成以下数据分析：")
        response_parts.append("")
        
        # 提取关键信息
        if "step1_user_profile" in tool_results:
            profile = tool_results["step1_user_profile"]
            if profile and isinstance(profile, dict):
                response_parts.append("👤 **用户档案**：")
                if "age" in profile:
                    response_parts.append(f"  - 年龄：{profile['age']}岁")
                if "primary_goal" in profile:
                    response_parts.append(f"  - 目标：{profile['primary_goal']}")
                if "fitness_level" in profile:
                    response_parts.append(f"  - 水平：{profile['fitness_level']}")
                response_parts.append("")
        
        if "step4_complexity" in tool_results:
            complexity = tool_results["step4_complexity"]
            if complexity and isinstance(complexity, dict):
                is_complex = complexity.get("is_complex", False)
                response_parts.append(f"🔍 **查询分析**：{'复杂' if is_complex else '简单'}查询")
                response_parts.append("")
        
        if "step8_retrieval_results" in tool_results:
            retrieval = tool_results["step8_retrieval_results"]
            if retrieval and isinstance(retrieval, dict):
                count = retrieval.get("count", 0)
                response_parts.append(f"📊 **检索结果**：找到 {count} 个相关推荐")
                response_parts.append("")
    
    # 添加建议
    response_parts.extend([
        "💡 **建议**：",
        "  - 请稍后重试获取AI分析",
        "  - 或联系客服获取人工指导",
        "  - 您也可以查看上述数据自行分析",
        "",
        "感谢您的理解！"
    ])
    
    return "\n".join(response_parts)


async def stream_deepseek(
    query: str,
    few_shot_examples: List[Dict[str, Any]],
    tool_results: Dict[str, Any],
    system_prompt: str = "你是一位专业的健身教练。"
) -> AsyncIterator[str]:
    """
    流式调用DeepSeek API

    Args:
        query: 用户查询
        few_shot_examples: Few-Shot示例列表
        tool_results: 工具调用结果
        system_prompt: 系统提示词

    Yields:
        str: 流式返回的文本片段
    """
    # 检查API密钥
    if not LLMConfig.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY未配置，请在.env文件中配置")

    try:
        # 构建消息（同步版本）
        messages = [{"role": "system", "content": system_prompt}]

        for example in few_shot_examples:
            messages.append({"role": "user", "content": example.get("query", "")})
            messages.append({"role": "assistant", "content": example.get("response", "")})

        tool_context = _format_tool_results(tool_results)
        if tool_context:
            messages.append({"role": "system", "content": f"工具调用结果:\n{tool_context}"})

        messages.append({"role": "user", "content": query})

        # 流式调用
        async with httpx.AsyncClient(timeout=LLMConfig.TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{LLMConfig.DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLMConfig.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLMConfig.DEEPSEEK_MODEL,
                    "messages": messages,
                    "max_tokens": LLMConfig.MAX_TOKENS,
                    "temperature": LLMConfig.TEMPERATURE,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # 去掉 "data: " 前缀

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]

                            if "content" in delta:
                                yield delta["content"]
                        except Exception as e:
                            logger.warning(f"解析流式响应失败: {e}")
                            continue

    except Exception as e:
        # 记录详细的错误上下文
        logger.error(
            f"DeepSeek流式调用失败: {e}\n"
            f"上下文信息:\n"
            f"  - 模型: {LLMConfig.DEEPSEEK_MODEL}\n"
            f"  - 查询: {query[:100]}...\n"
            f"  - Few-Shot示例数: {len(few_shot_examples)}\n"
            f"  - 系统提示词长度: {len(system_prompt)}\n"
            f"  - 流式模式: True",
            exc_info=True
        )
        raise
