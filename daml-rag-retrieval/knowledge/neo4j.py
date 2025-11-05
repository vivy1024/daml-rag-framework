#!/usr/bin/env python3
"""
DAML-RAG 框架 Neo4j知识图谱实现
专业图数据库支持，用于结构化关系推理
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import json
from datetime import datetime

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, TransientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from ..base import BaseKnowledgeGraphRetriever, GraphResult

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60
    connection_timeout: int = 30
    max_transaction_retry_time: int = 30
    initial_retry_delay_ms: int = 1000
    retry_delay_multiplier: float = 2.0
    driver_config: Optional[Dict[str, Any]] = None


class Neo4jKnowledgeRetriever(BaseKnowledgeGraphRetriever):
    """Neo4j知识图谱检索器"""

    def __init__(self, config: Neo4jConfig):
        super().__init__(config)
        self.config = config
        self.driver: Optional[AsyncGraphDatabase.driver] = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化Neo4j连接"""
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j driver not available. Install with: pip install neo4j"
            )

        try:
            # 配置驱动参数
            driver_config = self.config.driver_config or {}
            default_config = {
                "max_connection_lifetime": self.config.max_connection_lifetime,
                "max_connection_pool_size": self.config.max_connection_pool_size,
                "connection_acquisition_timeout": self.config.connection_acquisition_timeout,
                "connection_timeout": self.config.connection_timeout,
                "max_transaction_retry_time": self.config.max_transaction_retry_time,
                "initial_retry_delay_ms": self.config.initial_retry_delay_ms,
                "retry_delay_multiplier": self.config.retry_delay_multiplier,
                "trust": "TRUST_ALL_CERTIFICATES",
                "encrypted": False
            }
            driver_config.update(default_config)

            # 创建异步驱动
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                **driver_config
            )

            # 测试连接
            await self._test_connection()

            # 确保数据库存在
            await self._ensure_database_exists()

            self._initialized = True
            logger.info(f"Neo4j knowledge retriever initialized: {self.config.uri}")

        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise

    async def _test_connection(self) -> None:
        """测试Neo4j连接"""
        async with self.driver.session(database=self.config.database) as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            if record["test"] != 1:
                raise Exception("Neo4j connection test failed")

    async def _ensure_database_exists(self) -> None:
        """确保数据库存在"""
        try:
            async with self.driver.session() as session:
                # 列出所有数据库
                result = await session.run("SHOW DATABASES")
                databases = [record["name"] for record in result]

                if self.config.database not in databases:
                    # 创建数据库
                    await session.run(f"CREATE DATABASE {self.config.database}")
                    logger.info(f"Created database: {self.config.database}")

        except Exception as e:
            logger.warning(f"Could not verify database existence: {e}")

    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """执行Cypher查询"""
        if not self._initialized:
            await self.initialize()

        database = database or self.config.database
        parameters = parameters or {}

        try:
            async with self.driver.session(database=database) as session:
                result = await session.run(query, parameters)
                records = []
                async for record in result:
                    # 将记录转换为字典
                    record_dict = dict(record)
                    records.append(record_dict)
                return records

        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {query}, error: {e}")
            raise

    async def add_node(
        self,
        label: str,
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> str:
        """添加节点"""
        if not node_id:
            node_id = str(uuid.uuid4())

        # 添加ID属性
        properties["id"] = node_id

        # 构建Cypher查询
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        query = f"""
        CREATE (n:{label} {{{props_str}}})
        RETURN n.id as id
        """

        try:
            result = await self.execute_cypher(query, properties)
            if result:
                logger.debug(f"Added node: {label} with ID: {node_id}")
                return node_id
            else:
                raise Exception("Failed to add node")

        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    async def add_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """添加关系"""
        if not properties:
            properties = {}

        query = f"""
        MATCH (a), (b)
        WHERE a.id = $from_id AND b.id = $to_id
        CREATE (a)-[r:{relationship_type}]->(b)
        SET r += $props
        RETURN r
        """

        parameters = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "props": properties
        }

        try:
            result = await self.execute_cypher(query, parameters)
            success = len(result) > 0
            if success:
                logger.debug(f"Added relationship: {from_node_id} -> {to_node_id} ({relationship_type})")
            return success

        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False

    async def query_nodes(
        self,
        label: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查询节点"""
        query_parts = ["MATCH (n"]
        parameters = {}

        if label:
            query_parts.append(f":{label}")

        query_parts.append(")")

        # 添加WHERE条件
        where_conditions = []
        if properties:
            for i, (key, value) in enumerate(properties.items()):
                param_name = f"prop_{i}"
                where_conditions.append(f"n.{key} = ${param_name}")
                parameters[param_name] = value

        if where_conditions:
            query_parts.append(" WHERE " + " AND ".join(where_conditions))

        query_parts.append(" RETURN n")

        if limit:
            query_parts.append(f" LIMIT {limit}")
            parameters["limit"] = limit

        query = "".join(query_parts)

        try:
            result = await self.execute_cypher(query, parameters)
            # 提取节点数据
            nodes = []
            for record in result:
                node_data = dict(record["n"])
                nodes.append(node_data)
            return nodes

        except Exception as e:
            logger.error(f"Failed to query nodes: {e}")
            return []

    async def query_relationships(
        self,
        from_label: Optional[str] = None,
        to_label: Optional[str] = None,
        relationship_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """查询关系"""
        query_parts = ["MATCH (a"]
        parameters = {}

        if from_label:
            query_parts.append(f":{from_label}")

        query_parts.append(")")

        if relationship_type:
            query_parts.append(f"-[r:{relationship_type}")
        else:
            query_parts.append("-[r")

        if properties:
            prop_str = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            query_parts.append(f" {{{prop_str}}}")

        query_parts.append("]->(")

        if to_label:
            query_parts.append(f":{to_label}")

        query_parts.append(")")

        # 添加属性过滤
        if properties:
            for key, value in properties.items():
                parameters[key] = value

        query_parts.append(" RETURN a, r, b")

        if limit:
            query_parts.append(f" LIMIT {limit}")

        query = "".join(query_parts)

        try:
            result = await self.execute_cypher(query, parameters)
            # 提取关系数据
            relationships = []
            for record in result:
                rel_data = {
                    "from_node": dict(record["a"]),
                    "relationship": dict(record["r"]),
                    "to_node": dict(record["b"])
                }
                relationships.append(rel_data)
            return relationships

        except Exception as e:
            logger.error(f"Failed to query relationships: {e}")
            return []

    async def graph_traversal(
        self,
        start_node_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        max_depth: int = 3,
        relationship_types: Optional[List[str]] = None,
        node_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """图遍历查询"""
        if direction == "outgoing":
            match_pattern = f"(start)-[r*1..{max_depth}]->(end)"
        elif direction == "incoming":
            match_pattern = f"(start)<-[r*1..{max_depth}]-(end)"
        else:  # both
            match_pattern = f"(start)-[r*1..{max_depth}]-(end)"

        query_parts = [
            "MATCH (start) WHERE start.id = $start_id",
            f"MATCH {match_pattern}"
        ]

        parameters = {"start_id": start_node_id}

        # 添加关系类型过滤
        if relationship_types:
            rel_types_str = "|".join(relationship_types)
            query_parts[-1] = query_parts[-1].replace("[r*", f"[r:{rel_types_str}*")
            parameters["relationship_types"] = relationship_types

        # 添加节点过滤
        if node_filters:
            where_conditions = []
            for i, (key, value) in enumerate(node_filters.items()):
                param_name = f"filter_{i}"
                where_conditions.append(f"end.{key} = ${param_name}")
                parameters[param_name] = value

            if where_conditions:
                query_parts.append(" WHERE " + " AND ".join(where_conditions))

        query_parts.append(" RETURN DISTINCT end, length([r]) as depth ORDER BY depth")

        query = "\n".join(query_parts)

        try:
            result = await self.execute_cypher(query, parameters)
            # 提取遍历结果
            nodes = []
            for record in result:
                node_data = dict(record["end"])
                node_data["traversal_depth"] = record["depth"]
                nodes.append(node_data)
            return nodes

        except Exception as e:
            logger.error(f"Failed to perform graph traversal: {e}")
            return []

    async def shortest_path(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """最短路径查询"""
        query_parts = [
            "MATCH (start), (end)",
            "WHERE start.id = $from_id AND end.id = $to_id"
        ]

        if relationship_types:
            rel_types_str = "|".join(relationship_types)
            query_parts.append(
                f"MATCH p = shortestPath((start)-[*1..{max_depth}:{rel_types_str}]-(end))"
            )
        else:
            query_parts.append(
                f"MATCH p = shortestPath((start)-[*1..{max_depth}]-(end))"
            )

        query_parts.append("RETURN p")

        query = "\n".join(query_parts)

        parameters = {
            "from_id": from_node_id,
            "to_id": to_node_id
        }

        try:
            result = await self.execute_cypher(query, parameters)
            if result:
                path = result[0]["p"]
                # 提取路径中的节点和关系
                path_data = []
                for i, node in enumerate(path.nodes):
                    path_data.append({
                        "type": "node",
                        "step": i,
                        "data": dict(node)
                    })
                return path_data
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to find shortest path: {e}")
            return None

    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """更新节点属性"""
        query_parts = ["MATCH (n) WHERE n.id = $node_id"]

        # 构建SET子句
        set_clauses = []
        for key, value in properties.items():
            set_clauses.append(f"n.{key} = ${key}")

        if set_clauses:
            query_parts.append(" SET " + ", ".join(set_clauses))

        query_parts.append(" RETURN n")

        query = "\n".join(query_parts)

        parameters = {"node_id": node_id, **properties}

        try:
            result = await self.execute_cypher(query, parameters)
            success = len(result) > 0
            if success:
                logger.debug(f"Updated node: {node_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to update node: {e}")
            return False

    async def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        query = """
        MATCH (n) WHERE n.id = $node_id
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """

        try:
            result = await self.execute_cypher(query, {"node_id": node_id})
            success = len(result) > 0 and result[0]["deleted_count"] > 0
            if success:
                logger.debug(f"Deleted node: {node_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to delete node: {e}")
            return False

    async def get_node_count(self, label: Optional[str] = None) -> int:
        """获取节点数量"""
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"

        try:
            result = await self.execute_cypher(query)
            return result[0]["count"] if result else 0

        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return 0

    async def get_relationship_count(self, relationship_type: Optional[str] = None) -> int:
        """获取关系数量"""
        if relationship_type:
            query = f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) as count"

        try:
            result = await self.execute_cypher(query)
            return result[0]["count"] if result else 0

        except Exception as e:
            logger.error(f"Failed to get relationship count: {e}")
            return 0

    async def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        try:
            # 获取节点标签统计
            labels_query = """
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
            YIELD value
            RETURN label, value.count as count
            """

            # 获取关系类型统计
            rel_types_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {})
            YIELD value
            RETURN relationshipType, value.count as count
            """

            # 获取数据库概览
            overview_query = """
            MATCH (n) OPTIONAL MATCH ()-[r]->()
            RETURN count(DISTINCT n) as nodes, count(r) as relationships
            """

            labels_result = await self.execute_cypher(labels_query) if NEO4J_AVAILABLE else []
            rel_types_result = await self.execute_cypher(rel_types_query) if NEO4J_AVAILABLE else []
            overview_result = await self.execute_cypher(overview_query) if NEO4J_AVAILABLE else []

            return {
                "database": self.config.database,
                "uri": self.config.uri,
                "overview": overview_result[0] if overview_result else {},
                "node_labels": labels_result,
                "relationship_types": rel_types_result,
                "connection_status": "connected" if self._initialized else "disconnected"
            }

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.driver:
                return False

            async with self.driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")
                return True

        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    async def close(self):
        """关闭连接"""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")


class Neo4jKnowledgeManager:
    """Neo4j知识图谱管理器 - 提供高级管理功能"""

    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.retriever: Optional[Neo4jKnowledgeRetriever] = None

    async def initialize(self) -> None:
        """初始化管理器"""
        self.retriever = Neo4jKnowledgeRetriever(self.config)
        await self.retriever.initialize()

    async def create_schema(
        self,
        schema_definition: Dict[str, Any]
    ) -> bool:
        """创建图数据库模式"""
        if not self.retriever:
            await self.initialize()

        try:
            # 创建约束
            constraints = schema_definition.get("constraints", [])
            for constraint in constraints:
                constraint_type = constraint.get("type")
                label = constraint.get("label")
                property = constraint.get("property")

                if constraint_type == "unique":
                    query = f"CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.{property} IS UNIQUE"
                elif constraint_type == "exists":
                    query = f"CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.{property} IS NOT NULL"
                else:
                    continue

                await self.retriever.execute_cypher(query)
                logger.info(f"Created constraint: {query}")

            # 创建索引
            indexes = schema_definition.get("indexes", [])
            for index in indexes:
                label = index.get("label")
                properties = index.get("properties", [])

                if properties:
                    props_str = ", ".join(properties)
                    query = f"CREATE INDEX FOR (n:{label}) ON (n.{props_str})"
                    await self.retriever.execute_cypher(query)
                    logger.info(f"Created index: {query}")

            logger.info("Schema creation completed")
            return True

        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False

    async def batch_import(
        self,
        nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """批量导入数据"""
        if not self.retriever:
            await self.initialize()

        results = {"nodes_imported": 0, "relationships_imported": 0}

        # 批量导入节点
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            for node in batch:
                try:
                    await self.retriever.add_node(
                        label=node["label"],
                        properties=node["properties"],
                        node_id=node.get("id")
                    )
                    results["nodes_imported"] += 1
                except Exception as e:
                    logger.error(f"Failed to import node: {e}")

            logger.info(f"Batch {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1} nodes completed")

        # 批量导入关系
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            for rel in batch:
                try:
                    await self.retriever.add_relationship(
                        from_node_id=rel["from_id"],
                        to_node_id=rel["to_id"],
                        relationship_type=rel["type"],
                        properties=rel.get("properties", {})
                    )
                    results["relationships_imported"] += 1
                except Exception as e:
                    logger.error(f"Failed to import relationship: {e}")

            logger.info(f"Batch {i//batch_size + 1}/{(len(relationships)-1)//batch_size + 1} relationships completed")

        return results

    async def export_graph(
        self,
        output_file: str,
        format: str = "json"
    ) -> bool:
        """导出图数据"""
        if not self.retriever:
            await self.initialize()

        try:
            # 导出所有节点
            nodes = await self.retriever.query_nodes()

            # 导出所有关系
            relationships = await self.retriever.query_relationships()

            export_data = {
                "nodes": nodes,
                "relationships": relationships,
                "export_timestamp": datetime.now().isoformat(),
                "total_nodes": len(nodes),
                "total_relationships": len(relationships)
            }

            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Graph exported to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        if not self.retriever:
            await self.initialize()

        try:
            # 获取数据库信息
            db_info = await self.retriever.get_database_info()

            # 获取节点和关系统计
            total_nodes = await self.retriever.get_node_count()
            total_relationships = await self.retriever.get_relationship_count()

            # 计算图密度
            max_possible_relationships = total_nodes * (total_nodes - 1) // 2
            density = (total_relationships / max_possible_relationships) if max_possible_relationships > 0 else 0

            return {
                **db_info,
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "graph_density": round(density, 4),
                "health": await self.retriever.health_check()
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    async def close(self):
        """关闭管理器"""
        if self.retriever:
            await self.retriever.close()
            self.retriever = None