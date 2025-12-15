# -*- coding: utf-8 -*-
"""完整知识图谱系统

整合Neo4j图数据库和向量搜索，提供企业级知识图谱功能
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from .neo4j_manager import Neo4jManager
from .vector_search_engine import VectorSearchEngine, create_text_for_embedding

logger = logging.getLogger(__name__)


class KnowledgeGraphFull:
    """
    完整知识图谱系统
    
    特性：
    - Neo4j图存储
    - 向量语义搜索
    - 图推理和路径查询
    - 知识融合
    - 高级查询API
    """
    
    def __init__(
        self,
        # Neo4j配置
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        neo4j_database: str = "neo4j",
        # Qdrant配置
        qdrant_host: str = None,
        qdrant_port: int = 6333,
        qdrant_collection: str = "training_knowledge",
        vector_size: int = 1024,  # BAAI/bge-small-en-v1.5 生成384维向量
        # Embedding配置
        embedding_model: Optional[str] = None
    ):
        """
        初始化知识图谱系统

        Args:
            neo4j_uri: Neo4j连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: Neo4j数据库名
            qdrant_host: Qdrant主机
            qdrant_port: Qdrant端口
            qdrant_collection: Qdrant集合名
            vector_size: 向量维度
            embedding_model: Embedding模型名称
        """
        logger.info("=" * 80)
        logger.info("🚀 初始化完整知识图谱系统")
        logger.info("=" * 80)

        # 从环境变量读取配置（Docker环境）
        neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "build_body_2024")
        qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "qdrant")

        logger.info(f"Neo4j URI: {neo4j_uri}")
        logger.info(f"Qdrant Host: {qdrant_host}")

        # 初始化Neo4j
        self.neo4j = Neo4jManager(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database
        )

        # 初始化向量搜索
        self.vector_search = VectorSearchEngine(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=qdrant_collection,
            vector_size=vector_size,
            embedding_model=embedding_model
        )
        
        # 创建索引
        self.neo4j.create_indexes()
        
        logger.info("=" * 80)
        logger.info("✅ 知识图谱系统初始化完成")
        logger.info("=" * 80)
    
    # ==================== 基础操作 ====================
    
    def create_node_with_vector(
        self,
        label: str,
        properties: Dict[str, Any],
        text_for_embedding: Optional[str] = None
    ) -> str:
        """
        创建节点并生成向量
        
        Args:
            label: 节点标签
            properties: 节点属性
            text_for_embedding: 用于嵌入的文本（可选）
        
        Returns:
            节点ID
        """
        # 创建Neo4j节点
        node = self.neo4j.create_node(label, properties, return_node=True)
        
        if node:
            node_id = node.get("id", str(node.get("elementId")))
            
            # 生成向量
            if text_for_embedding is None:
                text_for_embedding = create_text_for_embedding(properties)
            
            vector = self.vector_search.encode(text_for_embedding)
            
            # 添加向量
            self.vector_search.add(
                id=node_id,
                vector=vector,
                payload={
                    "label": label,
                    **properties
                }
            )
            
            return node_id
        
        return None
    
    def batch_create_nodes_with_vectors(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        批量创建节点并生成向量
        
        Args:
            label: 节点标签
            nodes: 节点列表
            batch_size: 批次大小
        
        Returns:
            创建的节点数量
        """
        logger.info(f"📊 批量创建节点: {len(nodes)} 个 ({label})")
        
        # 批量创建Neo4j节点
        node_count = self.neo4j.batch_create_nodes(label, nodes, batch_size)
        
        # 生成文本和向量
        logger.info("🔤 生成向量...")
        texts = [create_text_for_embedding(node) for node in nodes]
        vectors = self.vector_search.encode(texts)
        
        # 批量添加向量
        logger.info("💾 批量添加向量...")
        ids = [node.get("id", f"{label}_{i}") for i, node in enumerate(nodes)]
        payloads = [{"label": label, **node} for node in nodes]
        
        self.vector_search.batch_add(ids, vectors, payloads, batch_size)
        
        logger.info(f"✅ 批量创建完成: {node_count} 个节点")
        return node_count
    
    # ==================== 查询操作 ====================
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_neighbors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 过滤条件
            include_neighbors: 是否包含邻居节点
        
        Returns:
            搜索结果列表
        """
        # 向量搜索
        vector_results = self.vector_search.search_by_text(query, top_k, filter)
        
        results = []
        for vr in vector_results:
            result = {
                "id": vr.id,
                "score": vr.score,
                "label": vr.payload.get("label"),
                "properties": vr.payload
            }
            
            # 获取邻居（如果需要）
            if include_neighbors:
                neighbors = self.neo4j.get_neighbors(vr.id, depth=1)
                result["neighbors"] = neighbors
            
            results.append(result)
        
        return results
    
    def find_path(
        self,
        from_node: str,
        to_node: str,
        max_depth: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        查找最短路径
        
        Args:
            from_node: 起始节点ID
            to_node: 目标节点ID
            max_depth: 最大深度
        
        Returns:
            路径信息
        """
        path = self.neo4j.find_shortest_path(from_node, to_node, max_depth)
        
        if path:
            return {
                "length": len(path["nodes"]) - 1,
                "nodes": path["nodes"],
                "relationships": path["relationships"]
            }
        
        return None
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        rel_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取邻居节点
        
        Args:
            node_id: 节点ID
            direction: 方向 ("in", "out", "both")
            rel_types: 关系类型列表
            depth: 深度
        
        Returns:
            邻居节点列表
        """
        return self.neo4j.get_neighbors(node_id, direction, rel_types, depth)
    
    def recommend_similar(
        self,
        node_id: str,
        top_k: int = 10,
        label_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        推荐相似节点
        
        Args:
            node_id: 节点ID
            top_k: 返回数量
            label_filter: 标签过滤
        
        Returns:
            相似节点列表
        """
        # 获取节点向量
        node_vector = self.vector_search.get_by_id(node_id)
        
        if not node_vector or node_vector.vector is None:
            logger.warning(f"⚠️ 节点 {node_id} 没有向量")
            return []
        
        # 向量搜索
        filter_dict = {"label": label_filter} if label_filter else None
        results = self.vector_search.search(
            node_vector.vector,
            top_k=top_k + 1,  # +1 因为可能包含自己
            filter=filter_dict
        )
        
        # 排除自己
        similar = [
            {
                "id": r.id,
                "score": r.score,
                "label": r.payload.get("label"),
                "properties": r.payload
            }
            for r in results
            if r.id != node_id
        ][:top_k]
        
        return similar
    
    # ==================== 高级查询 ====================
    
    def query_exercises_by_muscle(
        self,
        muscle_name: str,
        difficulty: Optional[str] = None,
        equipment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查询针对特定肌群的动作
        
        Args:
            muscle_name: 肌群名称
            difficulty: 难度
            equipment: 器械
        
        Returns:
            动作列表
        """
        # 构建查询
        where_clauses = ["m.name_zh CONTAINS $muscle_name OR m.name_zh = $muscle_name"]
        params = {"muscle_name": muscle_name}
        
        if difficulty:
            where_clauses.append("e.difficulty = $difficulty")
            params["difficulty"] = difficulty
        
        if equipment:
            where_clauses.append("e.equipment_zh CONTAINS $equipment")
            params["equipment"] = equipment
        
        where_str = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (e:Exercise)-[r:TARGETS_PRIMARY|TARGETS_SECONDARY]->(m:Muscle)
        WHERE {where_str}
        RETURN e, type(r) as rel_type, r.strength as strength
        ORDER BY r.strength DESC
        LIMIT 50
        """
        
        results = self.neo4j.execute_query(query, params)
        
        exercises = [
            {
                "id": r["e"].get("id"),
                "name": r["e"].get("name_zh"),
                "difficulty": r["e"].get("difficulty"),
                "equipment": r["e"].get("equipment_zh"),
                "relation_type": r["rel_type"],
                "strength": r.get("strength")
            }
            for r in results
        ]
        
        return exercises
    
    def query_nutrition_by_goal(
        self,
        goal: str,
        calories_range: Optional[Tuple[int, int]] = None,
        protein_min: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        根据目标查询营养食物
        
        Args:
            goal: 目标 ("增肌", "减脂", "维持")
            calories_range: 热量范围 (min, max)
            protein_min: 最小蛋白质含量
        
        Returns:
            食物列表
        """
        where_clauses = []
        params = {}
        
        if calories_range:
            where_clauses.append("f.calories >= $cal_min AND f.calories <= $cal_max")
            params["cal_min"] = calories_range[0]
            params["cal_max"] = calories_range[1]
        
        if protein_min:
            where_clauses.append("f.protein >= $protein_min")
            params["protein_min"] = protein_min
        
        where_str = " AND ".join(where_clauses) if where_clauses else "true"
        
        query = f"""
        MATCH (f:Food)
        WHERE {where_str}
        RETURN f
        ORDER BY f.protein DESC
        LIMIT 50
        """
        
        results = self.neo4j.execute_query(query, params)
        
        foods = [
            {
                "id": r["f"].get("id"),
                "name": r["f"].get("name_zh"),
                "calories": r["f"].get("calories"),
                "protein": r["f"].get("protein"),
                "carbs": r["f"].get("carbs"),
                "fat": r["f"].get("fat")
            }
            for r in results
        ]
        
        return foods
    
    def get_training_plan_by_level(
        self,
        level: str,
        goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        根据水平获取训练计划
        
        Args:
            level: 训练水平 ("beginner", "intermediate", "advanced")
            goal: 目标
        
        Returns:
            训练计划
        """
        # 查询周期化模型
        query = """
        MATCH (pm:PeriodizationModel)-[:HAS_PHASE]->(tp:TrainingPhase)
        WHERE $level IN pm.best_for
        RETURN pm, collect(tp) as phases
        LIMIT 1
        """
        
        results = self.neo4j.execute_query(query, {"level": level})
        
        if results:
            model = results[0]["pm"]
            phases = results[0]["phases"]
            
            return {
                "model": {
                    "name": model.get("name"),
                    "description": model.get("description"),
                    "advantages": model.get("advantages")
                },
                "phases": [
                    {
                        "name": phase.get("name"),
                        "weeks": phase.get("weeks"),
                        "intensity": phase.get("intensity"),
                        "sets": phase.get("sets"),
                        "reps": phase.get("reps")
                    }
                    for phase in phases
                ]
            }
        
        return {}
    
    # ==================== 统计和管理 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            统计信息
        """
        neo4j_stats = self.neo4j.get_statistics()
        vector_count = self.vector_search.count()
        
        return {
            **neo4j_stats,
            "vector_count": vector_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, bool]:
        """
        健康检查
        
        Returns:
            各组件健康状态
        """
        health = {}
        
        # Neo4j健康
        try:
            self.neo4j.execute_query("RETURN 1")
            health["neo4j"] = True
        except Exception as e:
            logger.error(f"❌ Neo4j健康检查失败: {e}")
            health["neo4j"] = False
        
        # Qdrant健康
        try:
            self.vector_search.count()
            health["qdrant"] = True
        except Exception as e:
            logger.error(f"❌ Qdrant健康检查失败: {e}")
            health["qdrant"] = False
        
        return health
    
    def close(self):
        """关闭连接"""
        self.neo4j.close()
        self.vector_search.close()
        logger.info("🔒 知识图谱系统已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

