# -*- coding: utf-8 -*-
"""
DAML-RAG图检索引擎 v2.0

实现基于知识图谱的关系推理检索：
- Neo4j图数据库集成
- Cypher查询构建和优化
- 路径发现和关系推理
- 图约束和安全性检查

版本：v2.0.0
更新日期：2025-11-17
设计原则：关系推理、路径发现、安全约束
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.retrieval import (
    IGraphRetriever, QueryRequest, RetrievalResult, RetrievalResponse,
    RetrievalMode, BaseRetriever
)
from ..interfaces.storage import IGraphStorage, GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """图查询类型"""
    ENTITY_SEARCH = "entity_search"         # 实体搜索
    RELATION_QUERY = "relation_query"       # 关系查询
    PATH_FINDING = "path_finding"          # 路径查找
    NEIGHBORHOOD = "neighborhood"          # 邻域查询
    PATTERN_MATCHING = "pattern_matching"  # 模式匹配
    CONSTRAINT_CHECK = "constraint_check"  # 约束检查


class RelationType(Enum):
    """关系类型"""
    IS_A = "IS_A"                    # 是一种（分类关系）
    PART_OF = "PART_OF"              # 部分属于（组成关系）
    RELATED_TO = "RELATED_TO"        # 相关关系
    CONTRAINDICATED = "CONTRAINDICATED"  # 禁忌关系
    RECOMMENDED_FOR = "RECOMMENDED_FOR"  # 推荐关系
    SYNERGY_WITH = "SYNERGY_WITH"   # 协同关系
    ALTERNATIVE_TO = "ALTERNATIVE_TO"  # 替代关系


@dataclass
class GraphQueryConfig:
    """图检索配置"""
    # 查询参数
    max_depth: int = 3                 # 最大搜索深度
    max_nodes: int = 100              # 最大节点数
    max_relationships: int = 500      # 最大关系数
    timeout_seconds: float = 30.0     # 查询超时时间

    # 路径查找参数
    max_paths: int = 10               # 最大路径数
    path_weight_strategy: str = "shortest"  # 路径权重策略

    # 过滤参数
    enable_safety_filter: bool = True   # 启用安全过滤
    enable_evidence_filter: bool = True  # 启用证据过滤
    min_evidence_level: float = 0.5     # 最小证据等级

    # 性能优化
    enable_result_cache: bool = True    # 启用结果缓存
    enable_query_optimization: bool = True  # 启用查询优化
    parallel_query_limit: int = 5       # 并行查询限制


@dataclass
class GraphQueryResult:
    """图查询结果"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    paths: List[List[str]]              # 发现的路径
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphRetriever(BaseRetriever, IGraphRetriever):
    """
    图检索引擎

    基于知识图谱的关系推理和约束验证。
    """

    def __init__(self, name: str = "GraphRetriever", version: str = "2.0.0"):
        super().__init__(name, version)
        self._graph_storage: Optional[IGraphStorage] = None
        self._config = GraphQueryConfig()
        self._query_cache = {}
        self._entity_index = {}          # 实体索引缓存
        self._relationship_index = {}    # 关系索引缓存

        # 性能指标
        self._metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_query_time': 0.0,
            'path_discoveries': 0,
            'constraint_validations': 0,
            'safety_blocks': 0
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化图检索引擎"""
        try:
            if config:
                self._update_config(config)

            # 初始化图存储
            if not self._graph_storage:
                logger.error("图存储未设置")
                return False

            # 连接图存储
            if not await self._graph_storage.connect():
                logger.error("图存储连接失败")
                return False

            # 构建索引
            await self._build_indexes()

            logger.info(f"✅ 图检索引擎初始化成功: {self.name}")
            self._state = self.ComponentState.READY
            return True

        except Exception as e:
            logger.error(f"❌ 图检索引擎初始化失败 {self.name}: {e}")
            self._state = self.ComponentState.ERROR
            return False

    def set_graph_storage(self, storage: IGraphStorage) -> None:
        """设置图存储"""
        self._graph_storage = storage

    async def _build_indexes(self) -> None:
        """构建索引"""
        try:
            logger.info("开始构建图索引...")

            # 构建实体索引
            await self._build_entity_index()

            # 构建关系索引
            await self._build_relationship_index()

            logger.info("✅ 图索引构建完成")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    async def _build_entity_index(self) -> None:
        """构建实体索引"""
        # 这里应该从图存储中加载所有实体并建立索引
        # 示例实现：
        # nodes = await self._graph_storage.get_all_nodes()
        # for node in nodes:
        #     self._entity_index[node.label.lower()] = node
        pass

    async def _build_relationship_index(self) -> None:
        """构建关系索引"""
        # 这里应该从图存储中加载所有关系并建立索引
        # 示例实现：
        # relationships = await self._graph_storage.get_all_relationships()
        # for rel in relationships:
        #     key = f"{rel.source_node}_{rel.type}_{rel.target_node}"
        #     self._relationship_index[key] = rel
        pass

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        if 'max_depth' in config:
            self._config.max_depth = config['max_depth']
        if 'max_nodes' in config:
            self._config.max_nodes = config['max_nodes']
        if 'enable_safety_filter' in config:
            self._config.enable_safety_filter = config['enable_safety_filter']
        if 'min_evidence_level' in config:
            self._config.min_evidence_level = config['min_evidence_level']

        logger.info(f"图检索配置已更新: {self.name}")

    async def retrieve(self, request: QueryRequest) -> RetrievalResponse:
        """执行图检索"""
        start_time = asyncio.get_event_loop().time()
        self._metrics['total_queries'] += 1

        try:
            # 1. 检查缓存
            if self._config.enable_result_cache:
                cached_result = self._get_from_cache(request)
                if cached_result:
                    self._metrics['cache_hits'] += 1
                    return cached_result

            # 2. 分析查询类型
            query_type = self._analyze_query_type(request)

            # 3. 提取实体和关系
            entities, relations = await self._extract_entities_and_relations(request.query_text)

            # 4. 执行图查询
            graph_results = await self._execute_graph_query(request, query_type, entities, relations)

            # 5. 后处理和验证
            validated_results = await self._validate_and_filter_results(graph_results, request)

            # 6. 构建响应
            execution_time = asyncio.get_event_loop().time() - start_time
            response = RetrievalResponse(
                query_id=request.query_id,
                results=validated_results,
                total_results=len(validated_results),
                execution_time=execution_time,
                metadata={
                    'query_type': query_type.value,
                    'extracted_entities': entities,
                    'extracted_relations': relations,
                    'graph_nodes': len(graph_results.nodes) if graph_results else 0,
                    'graph_relationships': len(graph_results.relationships) if graph_results else 0,
                    'discovered_paths': len(graph_results.paths) if graph_results else 0
                }
            )

            # 7. 缓存结果
            if self._config.enable_result_cache:
                self._cache_result(request, response)

            # 更新指标
            self._update_metrics(execution_time)

            logger.debug(f"图检索完成: {len(validated_results)} 结果, {execution_time:.3f}s")
            return response

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"图检索失败: {e}")

            return RetrievalResponse(
                query_id=request.query_id,
                results=[],
                total_results=0,
                execution_time=execution_time,
                error=str(e)
            )

    def _analyze_query_type(self, request: QueryRequest) -> QueryType:
        """分析查询类型"""
        query_text = request.query_text.lower()

        # 关系查询关键词
        relation_keywords = ['关系', '关联', '联系', '相关', '连接', '依赖']
        if any(keyword in query_text for keyword in relation_keywords):
            return QueryType.RELATION_QUERY

        # 路径查找关键词
        path_keywords = ['路径', '如何', '过程', '步骤', '流程', '序列']
        if any(keyword in query_text for keyword in path_keywords):
            return QueryType.PATH_FINDING

        # 邻域查询关键词
        neighborhood_keywords = ['附近', '周围', '相关', '相似', '类似']
        if any(keyword in query_text for keyword in neighborhood_keywords):
            return QueryType.NEIGHBORHOOD

        # 默认为实体搜索
        return QueryType.ENTITY_SEARCH

    async def _extract_entities_and_relations(self, query_text: str) -> Tuple[List[str], List[str]]:
        """提取实体和关系"""
        # 这里应该使用NLP技术提取实体和关系
        # 示例实现：
        entities = []
        relations = []

        # 简单的关键词匹配
        domain_entities = ['胸部', '腿部', '手臂', '背部', '腹部', '肩部']
        for entity in domain_entities:
            if entity in query_text:
                entities.append(entity)

        domain_relations = ['训练', '拉伸', '强化', '放松', '恢复']
        for relation in domain_relations:
            if relation in query_text:
                relations.append(relation)

        return entities, relations

    async def _execute_graph_query(
        self,
        request: QueryRequest,
        query_type: QueryType,
        entities: List[str],
        relations: List[str]
    ) -> GraphQueryResult:
        """执行图查询"""
        try:
            if query_type == QueryType.ENTITY_SEARCH:
                return await self._execute_entity_search(request, entities)
            elif query_type == QueryType.RELATION_QUERY:
                return await self._execute_relation_query(request, entities, relations)
            elif query_type == QueryType.PATH_FINDING:
                return await self._execute_path_finding(request, entities)
            elif query_type == QueryType.NEIGHBORHOOD:
                return await self._execute_neighborhood_query(request, entities)
            else:
                return await self._execute_pattern_matching(request, entities, relations)

        except Exception as e:
            logger.error(f"图查询执行失败: {e}")
            return GraphQueryResult(nodes=[], relationships=[], paths=[], score=0.0)

    async def _execute_entity_search(self, request: QueryRequest, entities: List[str]) -> GraphQueryResult:
        """执行实体搜索"""
        nodes = []
        relationships = []

        for entity in entities:
            # 查找匹配的节点
            matching_nodes = await self._graph_storage.search_nodes(
                properties={'name': entity, 'domain': request.domain}
            )
            nodes.extend(matching_nodes)

            # 查找相关的关系
            for node in matching_nodes:
                related_rels = await self._graph_storage.get_relationships(node.id)
                relationships.extend(related_rels)

        return GraphQueryResult(
            nodes=nodes,
            relationships=relationships,
            paths=[],
            score=len(nodes) / max(len(entities), 1)
        )

    async def _execute_relation_query(
        self,
        request: QueryRequest,
        entities: List[str],
        relations: List[str]
    ) -> GraphQueryResult:
        """执行关系查询"""
        nodes = []
        relationships = []
        paths = []

        # 查找指定类型的关系
        for relation in relations:
            matching_rels = await self._graph_storage.search_relationships(
                relationship_type=relation,
                properties={'domain': request.domain}
            )
            relationships.extend(matching_rels)

        # 收集相关节点
        node_ids = set()
        for rel in relationships:
            node_ids.add(rel.source_node)
            node_ids.add(rel.target_node)

        for node_id in node_ids:
            node = await self._graph_storage.get_node(node_id)
            if node:
                nodes.append(node)

        return GraphQueryResult(
            nodes=nodes,
            relationships=relationships,
            paths=paths,
            score=len(relationships) / max(len(relations), 1)
        )

    async def _execute_path_finding(self, request: QueryRequest, entities: List[str]) -> GraphQueryResult:
        """执行路径查找"""
        if len(entities) < 2:
            return GraphQueryResult(nodes=[], relationships=[], paths=[], score=0.0)

        paths = []
        nodes = []
        relationships = []

        # 查找实体间的路径
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                start_entity = entities[i]
                end_entity = entities[j]

                # 查找节点
                start_node = await self._find_node_by_name(start_entity)
                end_node = await self._find_node_by_name(end_entity)

                if start_node and end_node:
                    # 查找路径
                    found_paths = await self._graph_storage.find_path(
                        start_node.id, end_node.id, max_depth=self._config.max_depth
                    )
                    paths.extend(found_paths)

        self._metrics['path_discoveries'] += len(paths)

        return GraphQueryResult(
            nodes=nodes,
            relationships=relationships,
            paths=paths,
            score=len(paths) / max(len(entities), 1)
        )

    async def _execute_neighborhood_query(self, request: QueryRequest, entities: List[str]) -> GraphQueryResult:
        """执行邻域查询"""
        all_nodes = []
        all_relationships = []

        for entity in entities:
            node = await self._find_node_by_name(entity)
            if node:
                # 查找邻居节点
                neighbors = await self._graph_storage.get_neighbors(
                    node.id, max_depth=self._config.max_depth, max_nodes=self._config.max_nodes
                )
                all_nodes.extend(neighbors['nodes'])
                all_relationships.extend(neighbors['relationships'])

        return GraphQueryResult(
            nodes=all_nodes,
            relationships=all_relationships,
            paths=[],
            score=len(all_nodes) / max(len(entities), 1)
        )

    async def _execute_pattern_matching(
        self,
        request: QueryRequest,
        entities: List[str],
        relations: List[str]
    ) -> GraphQueryResult:
        """执行模式匹配"""
        # 这里可以实现更复杂的图模式匹配
        # 示例实现：简单的三元组匹配
        nodes = []
        relationships = []

        for entity in entities:
            node = await self._find_node_by_name(entity)
            if node:
                nodes.append(node)

                # 查找相关关系
                for relation in relations:
                    matching_rels = await self._graph_storage.search_relationships(
                        source_node=node.id,
                        relationship_type=relation
                    )
                    relationships.extend(matching_rels)

        return GraphQueryResult(
            nodes=nodes,
            relationships=relationships,
            paths=[],
            score=len(relationships) / max(len(entities) * len(relations), 1)
        )

    async def _find_node_by_name(self, name: str) -> Optional[GraphNode]:
        """根据名称查找节点"""
        # 先检查缓存
        if name.lower() in self._entity_index:
            return self._entity_index[name.lower()]

        # 从图存储查找
        nodes = await self._graph_storage.search_nodes(properties={'name': name})
        if nodes:
            node = nodes[0]
            self._entity_index[name.lower()] = node  # 缓存结果
            return node

        return None

    async def _validate_and_filter_results(
        self,
        graph_results: GraphQueryResult,
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """验证和过滤结果"""
        validated_results = []

        # 1. 安全性检查
        if self._config.enable_safety_filter:
            graph_results = await self._apply_safety_filter(graph_results)

        # 2. 证据等级检查
        if self._config.enable_evidence_filter:
            graph_results = await self._apply_evidence_filter(graph_results)

        # 3. 转换为RetrievalResult
        for node in graph_results.nodes:
            # 计算节点相关性分数
            relevance_score = self._calculate_relevance_score(node, request)

            if relevance_score > 0:
                result = RetrievalResult(
                    document_id=node.id,
                    content=node.properties.get('description', ''),
                    score=relevance_score,
                    metadata={
                        'node_label': node.label,
                        'node_type': node.properties.get('type', 'unknown'),
                        'evidence_level': node.properties.get('evidence_level', 0.0),
                        'safety_rating': node.properties.get('safety_rating', 'safe'),
                        'source': 'graph_retrieval'
                    }
                )
                validated_results.append(result)

        # 4. 按分数排序
        validated_results.sort(key=lambda x: x.score, reverse=True)

        # 5. 限制结果数量
        return validated_results[:request.top_k]

    async def _apply_safety_filter(self, graph_results: GraphQueryResult) -> GraphQueryResult:
        """应用安全过滤器"""
        safe_nodes = []
        safe_relationships = []

        # 过滤不安全的节点
        for node in graph_results.nodes:
            safety_rating = node.properties.get('safety_rating', 'safe')
            if safety_rating != 'unsafe':
                safe_nodes.append(node)
            else:
                self._metrics['safety_blocks'] += 1

        # 过滤不安全的关系
        for rel in graph_results.relationships:
            safety_rating = rel.properties.get('safety_rating', 'safe')
            if safety_rating != 'unsafe':
                safe_relationships.append(rel)

        return GraphQueryResult(
            nodes=safe_nodes,
            relationships=safe_relationships,
            paths=graph_results.paths,
            score=graph_results.score
        )

    async def _apply_evidence_filter(self, graph_results: GraphQueryResult) -> GraphQueryResult:
        """应用证据等级过滤器"""
        filtered_nodes = []
        filtered_relationships = []

        # 过滤低证据等级的节点
        for node in graph_results.nodes:
            evidence_level = node.properties.get('evidence_level', 0.0)
            if evidence_level >= self._config.min_evidence_level:
                filtered_nodes.append(node)

        # 过滤低证据等级的关系
        for rel in graph_results.relationships:
            evidence_level = rel.properties.get('evidence_level', 0.0)
            if evidence_level >= self._config.min_evidence_level:
                filtered_relationships.append(rel)

        return GraphQueryResult(
            nodes=filtered_nodes,
            relationships=filtered_relationships,
            paths=graph_results.paths,
            score=graph_results.score
        )

    def _calculate_relevance_score(self, node: GraphNode, request: QueryRequest) -> float:
        """计算节点相关性分数"""
        score = 0.0

        # 1. 领域匹配
        if node.properties.get('domain') == request.domain:
            score += 0.3

        # 2. 名称匹配
        node_name = node.properties.get('name', '').lower()
        query_words = request.query_text.lower().split()
        for word in query_words:
            if word in node_name:
                score += 0.2

        # 3. 证据等级
        evidence_level = node.properties.get('evidence_level', 0.0)
        score += evidence_level * 0.3

        # 4. 安全评级
        safety_rating = node.properties.get('safety_rating', 'safe')
        if safety_rating == 'safe':
            score += 0.2

        return min(score, 1.0)

    def _get_from_cache(self, request: QueryRequest) -> Optional[RetrievalResponse]:
        """从缓存获取结果"""
        cache_key = self._generate_cache_key(request)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        return None

    def _cache_result(self, request: QueryRequest, response: RetrievalResponse) -> None:
        """缓存查询结果"""
        cache_key = self._generate_cache_key(request)
        self._query_cache[cache_key] = response

    def _generate_cache_key(self, request: QueryRequest) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{request.query_text}_{request.domain}_{request.top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_metrics(self, execution_time: float) -> None:
        """更新性能指标"""
        total_queries = self._metrics['total_queries']
        current_avg = self._metrics['average_query_time']
        self._metrics['average_query_time'] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )

    async def get_supported_modes(self) -> List[RetrievalMode]:
        """获取支持的检索模式"""
        return [RetrievalMode.GRAPH]

    def get_metrics(self) -> Dict[str, Any]:
        """获取图检索指标"""
        base_metrics = super().get_metrics()
        return {
            **base_metrics,
            'total_queries': self._metrics['total_queries'],
            'cache_hits': self._metrics['cache_hits'],
            'cache_hit_rate': (
                self._metrics['cache_hits'] / max(self._metrics['total_queries'], 1)
            ),
            'average_query_time': self._metrics['average_query_time'],
            'path_discoveries': self._metrics['path_discoveries'],
            'constraint_validations': self._metrics['constraint_validations'],
            'safety_blocks': self._metrics['safety_blocks'],
            'cache_size': len(self._query_cache),
            'entity_index_size': len(self._entity_index),
            'relationship_index_size': len(self._relationship_index)
        }


# 导出
__all__ = [
    'GraphRetriever',
    'GraphQueryConfig',
    'GraphQueryResult',
    'QueryType',
    'RelationType'
]