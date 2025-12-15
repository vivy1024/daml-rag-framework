# -*- coding: utf-8 -*-
"""
Neo4j客户端

基于框架层的Neo4jManager提供统一的Neo4j客户端接口。
为应用层提供简化的图数据库访问方法。

作者: BUILD_BODY Team
版本: v2.0.0
日期: 2025-12-03
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .base_client import BaseClient, ClientConfig, ClientStatus
from ..retrieval.graph.neo4j_manager import Neo4jManager


@dataclass
class Neo4jClientConfig(ClientConfig):
    """Neo4j客户端配置"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "Neo4jClientConfig":
        """从环境变量创建配置"""
        import os
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )


class Neo4jClient(BaseClient):
    """
    Neo4j图数据库客户端

    提供简化的Neo4j操作接口：
    - 连接管理
    - Cypher查询执行
    - 事务支持
    - 错误处理
    """

    def __init__(self, config: Optional[Neo4jClientConfig] = None):
        """
        初始化Neo4j客户端

        Args:
            config: Neo4j客户端配置。如果为None，将从环境变量读取配置
        """
        self.neo4j_config = config or Neo4jClientConfig.from_env()
        super().__init__(self.neo4j_config)

        self._manager: Optional[Neo4jManager] = None

    async def connect(self) -> bool:
        """
        建立Neo4j连接

        Returns:
            bool: 连接是否成功
        """
        try:
            self.status = ClientStatus.CONNECTING

            # 创建Neo4j管理器
            self._manager = Neo4jManager(
                uri=self.neo4j_config.uri,
                user=self.neo4j_config.user,
                password=self.neo4j_config.password,
                database=self.neo4j_config.database,
                max_connection_lifetime=self.neo4j_config.max_connection_lifetime,
                max_connection_pool_size=self.neo4j_config.max_connection_pool_size,
                connection_timeout=self.neo4j_config.connection_timeout
            )

            self.status = ClientStatus.CONNECTED
            self.logger.info(f"✅ Neo4j客户端已连接: {self.neo4j_config.uri}")
            return True

        except Exception as e:
            self.status = ClientStatus.ERROR
            self.logger.error(f"❌ Neo4j客户端连接失败: {str(e)}")
            return False

    async def disconnect(self):
        """断开Neo4j连接"""
        if self._manager:
            self._manager.close()
            self._manager = None

        self.status = ClientStatus.DISCONNECTED
        self.logger.info("🔌 Neo4j客户端已断开连接")

    async def _execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Neo4j查询

        Args:
            request: 请求数据，包含query、parameters等

        Returns:
            Dict[str, Any]: 查询结果
        """
        if not self._manager:
            raise RuntimeError("Neo4j客户端未初始化")

        query = request["query"]
        parameters = request.get("parameters", {})

        # 执行查询
        result = self._manager.execute_query(query, parameters)

        return {
            "data": result,
            "query": query,
            "parameters": parameters
        }

    # 便捷查询方法
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行Cypher查询

        Args:
            query: Cypher查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        if not self._manager:
            raise RuntimeError("Neo4j客户端未连接")
        
        # 直接调用Neo4jManager执行查询
        result = self._manager.execute_query(query, parameters or {})
        return result

    async def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行写操作查询（创建、更新、删除）

        Args:
            query: Cypher查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        if not self._manager:
            raise RuntimeError("Neo4j客户端未初始化")

        parameters = parameters or {}

        with self._manager.transaction() as tx:
            result = tx.run(query, parameters)
            return [record.data() for record in result]

    async def find_nodes(self, label: str, properties: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        查找节点

        Args:
            label: 节点标签
            properties: 节点属性

        Returns:
            节点列表
        """
        if properties:
            # 构建属性条件
            conditions = []
            for key, value in properties.items():
                if isinstance(value, str):
                    conditions.append(f"n.{key} = '{value}'")
                else:
                    conditions.append(f"n.{key} = {value}")

            where_clause = " AND ".join(conditions)
            query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n"
        else:
            query = f"MATCH (n:{label}) RETURN n"

        result = await self.execute_query(query)
        return [record["n"] for record in result]

    async def find_relationships(
        self,
        start_label: Optional[str] = None,
        rel_type: Optional[str] = None,
        end_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查找关系

        Args:
            start_label: 起始节点标签
            rel_type: 关系类型
            end_label: 结束节点标签

        Returns:
            关系列表
        """
        match_parts = ["MATCH"]

        if start_label:
            start_node = "(s)"
        else:
            start_node = "(s)"

        if rel_type:
            rel_part = f"-[r:{rel_type}]-"
        else:
            rel_part = "-[r]-"

        if end_label:
            end_node = "(e)"
        else:
            end_node = "(e)"

        match_clause = f"{match_parts} {start_node}{rel_part}{end_node}"
        query = f"{match_clause} RETURN s, r, e"

        result = await self.execute_query(query)
        return [
            {
                "start": record["s"],
                "relationship": record["r"],
                "end": record["e"]
            }
            for record in result
        ]

    async def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建节点

        Args:
            label: 节点标签
            properties: 节点属性

        Returns:
            创建的节点
        """
        # 构建属性字符串
        props_str = ", ".join([f"{k}: {self._format_value(v)}" for k, v in properties.items()])
        query = f"CREATE (n:{label} {{{props_str}}}) RETURN n"

        result = await self.execute_query(query)
        return result[0]["n"] if result else None

    async def create_relationship(
        self,
        start_node_id: str,
        rel_type: str,
        end_node_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建关系

        Args:
            start_node_id: 起始节点ID
            rel_type: 关系类型
            end_node_id: 结束节点ID
            properties: 关系属性

        Returns:
            创建的关系
        """
        props = properties or {}
        props_str = ", ".join([f"{k}: {self._format_value(v)}" for k, v in props.items()])

        if props_str:
            query = f"""
            MATCH (a), (b)
            WHERE ID(a) = {start_node_id} AND ID(b) = {end_node_id}
            CREATE (a)-[r:{rel_type} {{{props_str}}}]->(b)
            RETURN r
            """
        else:
            query = f"""
            MATCH (a), (b)
            WHERE ID(a) = {start_node_id} AND ID(b) = {end_node_id}
            CREATE (a)-[r:{rel_type}]->(b)
            RETURN r
            """

        result = await self.execute_query(query)
        return result[0]["r"] if result else None

    def _format_value(self, value: Any) -> str:
        """格式化Cypher值"""
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            items = [self._format_value(item) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            items = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return f"'{str(value)}'"

    async def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        queries = [
            {"name": "node_count", "query": "MATCH (n) RETURN count(n) as count"},
            {"name": "relationship_count", "query": "MATCH ()-[r]->() RETURN count(r) as count"},
            {"name": "labels", "query": "CALL db.labels() YIELD label RETURN collect(label) as labels"},
            {"name": "relationship_types", "query": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"}
        ]

        info = {}
        for query_info in queries:
            try:
                result = await self.execute_query(query_info["query"])
                if query_info["name"] in ["node_count", "relationship_count"]:
                    info[query_info["name"]] = result[0]["count"]
                else:
                    info[query_info["name"]] = result[0][query_info["name"]]
            except Exception as e:
                self.logger.warning(f"获取数据库信息失败: {query_info['name']}, 错误: {e}")
                info[query_info["name"]] = None

        return info

    @asynccontextmanager
    async def transaction(self):
        """事务上下文管理器"""
        if not self._manager:
            raise RuntimeError("Neo4j客户端未初始化")

        with self._manager.transaction() as tx:
            yield tx


# 便捷工厂函数
def create_neo4j_client(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j"
) -> Neo4jClient:
    """
    创建Neo4j客户端

    Args:
        uri: Neo4j连接URI
        user: 用户名
        password: 密码
        database: 数据库名称

    Returns:
        Neo4jClient: 配置好的Neo4j客户端
    """
    config = Neo4jClientConfig(
        uri=uri,
        user=user,
        password=password,
        database=database
    )

    return Neo4jClient(config)