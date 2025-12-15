# -*- coding: utf-8 -*-
"""
DAML-RAG框架统一客户端系统

提供与外部服务通信的统一接口，包括：
- HTTP客户端（用于REST API）
- MCP客户端（用于MCP协议通信）
- Neo4j客户端（用于图数据库访问）
- Backend客户端（用于后端API通信）
- LLM客户端（用于大语言模型调用）

作者: BUILD_BODY Team
版本: v2.1.0
日期: 2025-12-12
"""

from .base_client import BaseClient, ClientConfig
from .http_client import HTTPClient
from .mcp_client_v2 import ConfigurableMCPClient, create_configurable_mcp_client
from .neo4j_client import Neo4jClient
from .llm_client import call_deepseek, call_ollama, call_moonshot, LLMConfig, get_fallback_response

__all__ = [
    'BaseClient',
    'ClientConfig',
    'HTTPClient',
    'ConfigurableMCPClient',
    'create_configurable_mcp_client',
    'Neo4jClient',
    'call_deepseek',
    'call_ollama',
    'call_moonshot',
    'LLMConfig',
    'get_fallback_response'
]