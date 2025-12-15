# -*- coding: utf-8 -*-
"""
企业级三层检索引擎 - DAML-RAG框架核心组件

基于GraphRAG v3简洁架构,增强Neo4j直接连接能力,实现真正的三层检索。

三层架构:
- Layer 1: 向量语义检索 (Qdrant via GraphRAG API)
- Layer 2: 图谱关系推理 (Neo4j Direct Connection with Fallback)
- Layer 3: 专业规则约束 (Business Rules Engine)

设计原则:
1. 连接池管理 - 企业级Neo4j连接管理
2. 优雅降级 - Neo4j失败时自动降级到API
3. 清晰分层 - 每层职责明确,互不耦合
4. 完善监控 - 详细日志和性能指标

版本: v2.0.0
日期: 2025-11-25
作者: 薛小川
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager
import aiohttp

logger = logging.getLogger(__name__)


# ============ 数据类定义 ============

@dataclass
class LayerExecutionResult:
    """单层检索执行结果"""
    layer_name: str
    success: bool
    results: List[Dict[str, Any]]
    execution_time_ms: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ThreeLayerResult:
    """三层检索最终结果"""
    query: str
    domain: str
    final_results: List[Dict[str, Any]]
    layer_1_result: LayerExecutionResult
    layer_2_result: LayerExecutionResult
    layer_3_result: LayerExecutionResult
    total_confidence: float
    total_execution_time_ms: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_recommendations(self) -> List[Dict[str, Any]]:
        """Framework层兼容性属性"""
        return self.final_results


# ============ Neo4j连接管理器 ============

class Neo4jConnectionManager:
    """
    企业级Neo4j连接管理器

    功能:
    1. 连接池管理
    2. 健康检查
    3. 自动重连
    4. 优雅关闭
    """

    def __init__(
        self,
        uri: str = "bolt://neo4j:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0
    ):
        """初始化Neo4j连接管理器"""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.is_connected = False
        self.last_health_check = None

        # 连接池配置
        self.config = {
            "max_connection_lifetime": max_connection_lifetime,
            "max_connection_pool_size": max_connection_pool_size,
            "connection_timeout": connection_timeout
        }

        logger.info(f"Neo4j连接管理器已创建 - URI: {uri}, User: {user}, Password: {'***' if password else 'None'}")

    def connect(self) -> bool:
        """建立Neo4j连接"""
        try:
            from neo4j import GraphDatabase

            # 创建驱动实例
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password) if self.password else None,
                max_connection_lifetime=self.config["max_connection_lifetime"],
                max_connection_pool_size=self.config["max_connection_pool_size"],
                connection_timeout=self.config["connection_timeout"]
            )

            # 验证连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.is_connected = True
                    self.last_health_check = datetime.now()
                    logger.info("✅ Neo4j连接成功建立")
                    return True
                else:
                    logger.error("❌ Neo4j连接验证失败")
                    return False

        except ImportError:
            logger.error("❌ Neo4j驱动未安装: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"❌ Neo4j连接失败: {e}")
            self.is_connected = False
            return False

    @contextmanager
    def get_session(self):
        """获取Neo4j会话 (上下文管理器)"""
        if not self.is_connected or not self.driver:
            raise RuntimeError("Neo4j未连接,请先调用connect()")

        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.driver:
                return False

            with self.driver.session() as session:
                result = session.run("RETURN 1 AS health")
                result.single()
                self.last_health_check = datetime.now()
                return True
        except Exception as e:
            logger.warning(f"Neo4j健康检查失败: {e}")
            self.is_connected = False
            return False

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            logger.info("Neo4j连接已关闭")


# ============ 三层检索引擎 ============

class TrueThreeLayerEngine:
    """
    企业级三层检索引擎

    核心特性:
    1. Layer 1: Qdrant向量检索 (通过GraphRAG API)
    2. Layer 2: Neo4j图谱推理 (直接连接 + API备份)
    3. Layer 3: 业务规则验证 (Python逻辑)

    设计亮点:
    - 连接池管理
    - 优雅降级
    - 并行执行
    - 完善监控
    """

    def __init__(
        self,
        graphrag_api_port: str = "8001",
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        enable_neo4j_direct: bool = True,
        enable_parallel_execution: bool = False  # Layer 1和2不能并行,因为2依赖1
    ):
        """初始化三层检索引擎"""
        # API配置
        self.graphrag_api_port = graphrag_api_port or os.getenv('API_PORT', '8001')
        self.graphrag_api_base = f"http://localhost:{self.graphrag_api_port}/api/graphrag"

        # Neo4j配置
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')  # 无密码认证

        # 功能开关
        self.enable_neo4j_direct = enable_neo4j_direct
        self.enable_parallel_execution = enable_parallel_execution

        # Neo4j连接管理器
        self.neo4j_manager: Optional[Neo4jConnectionManager] = None
        self.neo4j_available = False

        # 性能统计
        self.stats = {
            "total_queries": 0,
            "layer1_success": 0,
            "layer2_neo4j_direct": 0,
            "layer2_api_fallback": 0,
            "layer3_success": 0,
            "total_errors": 0
        }

        logger.info(f"三层检索引擎已创建 - GraphRAG API: {self.graphrag_api_base}")

        # 初始化Neo4j连接
        if self.enable_neo4j_direct:
            self._initialize_neo4j_connection()

    def _initialize_neo4j_connection(self):
        """初始化Neo4j直连"""
        try:
            self.neo4j_manager = Neo4jConnectionManager(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password
            )

            # 尝试连接
            if self.neo4j_manager.connect():
                self.neo4j_available = True
                logger.info("✅ Neo4j直连已启用")
            else:
                logger.warning("⚠️ Neo4j直连失败,将使用API降级")
                self.neo4j_available = False

        except Exception as e:
            logger.error(f"❌ Neo4j连接管理器初始化失败: {e}")
            self.neo4j_available = False

    async def execute_three_layer_query(
        self,
        query: str,
        domain: str = "fitness_exercises",
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        safety_check: bool = True
    ) -> ThreeLayerResult:
        """
        执行完整的三层检索

        Args:
            query: 用户查询文本
            domain: 检索领域
            user_id: 用户ID
            user_profile: 用户档案
            filters: 过滤条件
            top_k: 返回结果数
            safety_check: 是否执行安全检查

        Returns:
            ThreeLayerResult: 三层检索结果
        """
        start_time = datetime.now()
        self.stats["total_queries"] += 1

        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"🔍 开始三层检索: {query}")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        try:
            # ============ Layer 1: 向量语义检索 ============
            layer1_result = await self._execute_layer1_vector_search(
                query=query,
                domain=domain,
                top_k=top_k * 3,  # Layer 1召回更多候选
                filters=filters,
                user_id=user_id
            )

            if not layer1_result.success or not layer1_result.results:
                logger.warning("Layer 1未返回结果,终止检索")
                return self._build_final_result(
                    query=query,
                    domain=domain,
                    layer1=layer1_result,
                    layer2=self._empty_layer_result("Layer2-Graph"),
                    layer3=self._empty_layer_result("Layer3-Rules"),
                    start_time=start_time
                )

            # ============ Layer 2: 图谱关系推理 ============
            layer2_result = await self._execute_layer2_graph_reasoning(
                query=query,
                domain=domain,
                vector_results=layer1_result.results,
                top_k=top_k * 2,
                user_id=user_id
            )

            # 选择使用哪层的结果进入Layer 3
            if layer2_result.success and layer2_result.results:
                candidates_for_layer3 = layer2_result.results
            else:
                logger.warning("Layer 2未返回结果,使用Layer 1结果")
                candidates_for_layer3 = layer1_result.results[:top_k * 2]

            # ============ Layer 3: 业务规则验证 ============
            layer3_result = await self._execute_layer3_business_rules(
                query=query,
                candidates=candidates_for_layer3,
                user_profile=user_profile,
                top_k=top_k,
                safety_check=safety_check
            )

            # ============ 构建最终结果 ============
            final_result = self._build_final_result(
                query=query,
                domain=domain,
                layer1=layer1_result,
                layer2=layer2_result,
                layer3=layer3_result,
                start_time=start_time
            )

            logger.info(f"✅ 三层检索完成: {len(final_result.final_results)}个结果, 耗时{final_result.total_execution_time_ms:.0f}ms")

            return final_result

        except Exception as e:
            logger.error(f"❌ 三层检索失败: {e}", exc_info=True)
            self.stats["total_errors"] += 1

            # 返回错误结果
            return ThreeLayerResult(
                query=query,
                domain=domain,
                final_results=[],
                layer_1_result=self._empty_layer_result("Layer1-Vector", error=str(e)),
                layer_2_result=self._empty_layer_result("Layer2-Graph"),
                layer_3_result=self._empty_layer_result("Layer3-Rules"),
                total_confidence=0.0,
                total_execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                reasoning=f"检索过程出错: {str(e)}",
                metadata={"error": str(e)}
            )

    async def execute_three_layer_search(
        self,
        query: str,
        domain: str = "fitness",
        user_id: str = None,
        context: Dict[str, Any] = None
    ):
        """
        Framework层适配器方法

        将framework层的调用转换为内部的execute_three_layer_query调用
        保持向后兼容性
        """
        # 提取参数
        user_profile = context.get("user_profile") if context else None
        filters = context.get("filters") if context else None
        top_k = context.get("top_k", 10) if context else 10
        safety_check = context.get("safety_check", True) if context else True

        # 调用原始方法
        return await self.execute_three_layer_query(
            query=query,
            domain=domain,
            user_id=user_id,
            user_profile=user_profile,
            filters=filters,
            top_k=top_k,
            safety_check=safety_check
        )

    async def _execute_layer1_vector_search(
        self,
        query: str,
        domain: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> LayerExecutionResult:
        """
        Layer 1: 向量语义检索 (Qdrant via GraphRAG API)
        """
        start_time = datetime.now()
        logger.info("→ Layer 1: 向量语义检索 (Qdrant)")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.graphrag_api_base}/query",
                    json={
                        "query_text": query,
                        "domain": domain,
                        "query_type": "semantic_search",
                        "top_k": top_k,
                        "filters": filters or {},
                        "return_reason": False,
                        "user_id": user_id or "anonymous"
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("data", {}).get("results", [])

                        # 计算置信度
                        if results:
                            avg_score = sum(r.get("score", 0) for r in results) / len(results)
                            confidence = min(avg_score, 1.0)
                        else:
                            confidence = 0.0

                        execution_time = (datetime.now() - start_time).total_seconds() * 1000
                        self.stats["layer1_success"] += 1

                        logger.info(f"  ✓ Layer 1完成: {len(results)}个向量结果, 置信度{confidence:.2f}")

                        return LayerExecutionResult(
                            layer_name="Layer1-Vector",
                            success=True,
                            results=results,
                            execution_time_ms=execution_time,
                            confidence=confidence,
                            metadata={
                                "source": "qdrant_via_api",
                                "count": len(results),
                                "avg_score": confidence
                            }
                        )
                    else:
                        error_msg = f"GraphRAG API返回错误: {response.status}"
                        logger.error(f"  ✗ {error_msg}")
                        return self._empty_layer_result("Layer1-Vector", error=error_msg)

        except asyncio.TimeoutError:
            error_msg = "Layer 1超时"
            logger.error(f"  ✗ {error_msg}")
            return self._empty_layer_result("Layer1-Vector", error=error_msg)
        except Exception as e:
            error_msg = f"Layer 1异常: {e}"
            logger.error(f"  ✗ {error_msg}")
            return self._empty_layer_result("Layer1-Vector", error=error_msg)

    async def _execute_layer2_graph_reasoning(
        self,
        query: str,
        domain: str,
        vector_results: List[Dict[str, Any]],
        top_k: int,
        user_id: Optional[str] = None
    ) -> LayerExecutionResult:
        """
        Layer 2: 图谱关系推理 (Neo4j Direct + API Fallback)
        """
        start_time = datetime.now()
        logger.info("→ Layer 2: 图谱关系推理 (Neo4j)")

        # 策略1: 尝试Neo4j直连
        if self.neo4j_available and self.neo4j_manager:
            neo4j_result = await self._query_neo4j_direct(query, vector_results, top_k)
            if neo4j_result.success:
                self.stats["layer2_neo4j_direct"] += 1
                return neo4j_result
            else:
                logger.warning("  ⚠️ Neo4j直连失败,降级到API")

        # 策略2: 降级到GraphRAG API
        api_result = await self._query_neo4j_via_api(query, domain, vector_results, top_k, user_id)
        if api_result.success:
            self.stats["layer2_api_fallback"] += 1

        return api_result

    async def _query_neo4j_direct(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int
    ) -> LayerExecutionResult:
        """通过Neo4j直连查询图谱"""
        start_time = datetime.now()

        try:
            # 从查询中提取肌肉关键词
            muscle_keywords = self._extract_muscle_keywords(query)
            graph_results = []

            if muscle_keywords and self.neo4j_manager:
                with self.neo4j_manager.get_session() as session:
                    for muscle in muscle_keywords[:3]:  # 限制关键词数量
                        cypher_query = """
                        MATCH (m:Muscle)
                        WHERE m.name_zh CONTAINS $muscle
                           OR m.name_en CONTAINS $muscle
                           OR m.name CONTAINS $muscle
                        MATCH (e:Exercise)-[r:TARGETS_PRIMARY|TARGETS_SECONDARY]->(m)
                        RETURN
                            e.name_zh AS exercise_zh,
                            e.name AS exercise_en,
                            e.difficulty AS difficulty,
                            e.equipment AS equipment,
                            m.name_zh AS muscle_name,
                            type(r) AS relationship_type,
                            m.mev AS mev,
                            m.mav AS mav,
                            m.mrv AS mrv
                        LIMIT $limit
                        """

                        result = session.run(
                            cypher_query,
                            muscle=muscle,
                            limit=top_k
                        )

                        for record in result:
                            graph_results.append({
                                "exercise_name_zh": record.get("exercise_zh", ""),
                                "exercise_name_en": record.get("exercise_en", ""),
                                "difficulty": record.get("difficulty", ""),
                                "equipment": record.get("equipment", ""),
                                "target_muscle": record.get("muscle_name", ""),
                                "relationship_type": record.get("relationship_type", ""),
                                "training_volume": {
                                    "mev": record.get("mev"),
                                    "mav": record.get("mav"),
                                    "mrv": record.get("mrv")
                                },
                                "source": "neo4j_direct",
                                "score": 0.8  # Neo4j图查询默认高分
                            })

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            confidence = 0.9 if graph_results else 0.0

            logger.info(f"  ✓ Neo4j直连完成: {len(graph_results)}个图谱结果")

            return LayerExecutionResult(
                layer_name="Layer2-Graph",
                success=bool(graph_results),
                results=graph_results,
                execution_time_ms=execution_time,
                confidence=confidence,
                metadata={
                    "source": "neo4j_direct",
                    "count": len(graph_results),
                    "muscle_keywords": muscle_keywords
                }
            )

        except Exception as e:
            logger.error(f"  ✗ Neo4j直连查询失败: {e}")
            return self._empty_layer_result("Layer2-Graph", error=str(e))

    async def _query_neo4j_via_api(
        self,
        query: str,
        domain: str,
        vector_results: List[Dict[str, Any]],
        top_k: int,
        user_id: Optional[str] = None
    ) -> LayerExecutionResult:
        """通过GraphRAG API查询图谱(降级方案)"""
        start_time = datetime.now()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.graphrag_api_base}/query",
                    json={
                        "query_text": query,
                        "domain": domain,
                        "query_type": "hybrid",  # 混合查询包含图谱
                        "top_k": top_k,
                        "return_reason": True,
                        "user_id": user_id or "anonymous"
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("data", {}).get("results", [])

                        execution_time = (datetime.now() - start_time).total_seconds() * 1000
                        confidence = 0.7 if results else 0.0

                        logger.info(f"  ✓ API降级完成: {len(results)}个结果")

                        return LayerExecutionResult(
                            layer_name="Layer2-Graph",
                            success=bool(results),
                            results=results,
                            execution_time_ms=execution_time,
                            confidence=confidence,
                            metadata={
                                "source": "api_fallback",
                                "count": len(results)
                            }
                        )
                    else:
                        error_msg = f"API返回错误: {response.status}"
                        logger.error(f"  ✗ {error_msg}")
                        return self._empty_layer_result("Layer2-Graph", error=error_msg)

        except Exception as e:
            error_msg = f"API查询失败: {e}"
            logger.error(f"  ✗ {error_msg}")
            return self._empty_layer_result("Layer2-Graph", error=error_msg)

    async def _execute_layer3_business_rules(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        top_k: int,
        safety_check: bool
    ) -> LayerExecutionResult:
        """
        Layer 3: 业务规则验证

        规则:
        1. 用户档案匹配 (经验等级)
        2. 安全性检查 (禁忌症)
        3. 器械可用性
        4. 训练容量合理性
        """
        start_time = datetime.now()
        logger.info("→ Layer 3: 业务规则验证")

        try:
            validated_results = []
            user_profile = user_profile or {}

            for candidate in candidates:
                # 规则1: 经验等级匹配
                if not self._match_fitness_level(candidate, user_profile):
                    continue

                # 规则2: 安全性检查
                if safety_check:
                    if not self._validate_safety(candidate, user_profile):
                        continue

                # 规则3: 器械可用性
                if not self._check_equipment_availability(candidate, user_profile):
                    continue

                # 规则4: 训练容量合理性
                volume_score = self._assess_training_volume(candidate, user_profile)

                # 添加规则评分
                candidate["rule_validation_score"] = volume_score
                candidate["validation_passed"] = True

                validated_results.append(candidate)

                if len(validated_results) >= top_k:
                    break

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            confidence = 0.95 if validated_results else 0.0

            self.stats["layer3_success"] += 1
            logger.info(f"  ✓ Layer 3完成: {len(validated_results)}/{len(candidates)}个通过规则验证")

            return LayerExecutionResult(
                layer_name="Layer3-Rules",
                success=bool(validated_results),
                results=validated_results,
                execution_time_ms=execution_time,
                confidence=confidence,
                metadata={
                    "validated_count": len(validated_results),
                    "total_candidates": len(candidates),
                    "pass_rate": len(validated_results) / len(candidates) if candidates else 0
                }
            )

        except Exception as e:
            logger.error(f"  ✗ Layer 3失败: {e}")
            return self._empty_layer_result("Layer3-Rules", error=str(e))

    # ============ 业务规则方法 ============

    def _match_fitness_level(
        self,
        exercise: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> bool:
        """匹配健身经验等级"""
        if not user_profile:
            return True

        user_level = user_profile.get("fitness_level", "intermediate").lower()
        exercise_difficulty = (exercise.get("difficulty") or "intermediate").lower()

        # 等级映射
        level_hierarchy = {
            "beginner": ["beginner", "easy", "novice"],
            "intermediate": ["beginner", "intermediate", "moderate", "novice"],
            "advanced": ["intermediate", "advanced", "hard", "elite"]
        }

        allowed_difficulties = level_hierarchy.get(user_level, ["intermediate"])
        return any(diff in exercise_difficulty for diff in allowed_difficulties)

    def _validate_safety(
        self,
        exercise: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> bool:
        """安全性验证"""
        if not user_profile:
            return True

        # 检查禁忌症
        contraindications = exercise.get("contraindications", [])
        user_conditions = user_profile.get("medical_conditions", [])

        for condition in user_conditions:
            if condition in contraindications:
                logger.debug(f"安全过滤: {exercise.get('exercise_name_zh')} - 禁忌症 {condition}")
                return False

        # 年龄限制
        user_age = user_profile.get("age", 30)
        if user_age > 60:
            difficulty = (exercise.get("difficulty") or "").lower()
            if "advanced" in difficulty or "elite" in difficulty:
                logger.debug(f"安全过滤: {exercise.get('exercise_name_zh')} - 高龄不适合高难度")
                return False

        return True

    def _check_equipment_availability(
        self,
        exercise: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> bool:
        """检查器械可用性"""
        if not user_profile:
            return True

        available_equipment = user_profile.get("available_equipment", [])
        if not available_equipment:
            return True  # 未指定器械限制

        required_equipment = exercise.get("equipment", "")
        if not required_equipment:
            return True

        # 检查器械是否可用
        if required_equipment not in available_equipment and "全部" not in available_equipment:
            logger.debug(f"器械过滤: {exercise.get('exercise_name_zh')} - 需要 {required_equipment}")
            return False

        return True

    def _assess_training_volume(
        self,
        exercise: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> float:
        """评估训练容量合理性"""
        volume_data = exercise.get("training_volume", {})
        if not volume_data:
            return 0.8  # 无训练容量数据,给默认分

        mev = volume_data.get("mev", 0)
        mav = volume_data.get("mav", 0)
        mrv = volume_data.get("mrv", 0)

        # 基于MEV/MAV/MRV评分
        if mev and mav and mrv:
            # 完整数据,高分
            return 1.0
        elif mev or mav:
            # 部分数据,中分
            return 0.9
        else:
            # 无数据,低分
            return 0.7

    def _extract_muscle_keywords(self, query: str) -> List[str]:
        """从查询中提取肌肉关键词"""
        muscle_mapping = {
            "胸": ["胸大肌", "胸部", "Chest", "Pectoralis"],
            "背": ["背阔肌", "背部", "Back", "Latissimus"],
            "肩": ["三角肌", "肩部", "Shoulder", "Deltoid"],
            "臂": ["肱二头肌", "肱三头肌", "手臂", "Biceps", "Triceps"],
            "腿": ["股四头肌", "腘绳肌", "腿部", "Quadriceps", "Hamstrings"],
            "臀": ["臀大肌", "臀部", "Glutes"],
            "腹": ["腹直肌", "腹肌", "腹部", "Abs", "Rectus Abdominis"],
            "核心": ["核心", "Core"]
        }

        keywords = []
        query_lower = query.lower()

        for key, muscles in muscle_mapping.items():
            if key in query or any(m.lower() in query_lower for m in muscles):
                keywords.extend(muscles)

        return list(set(keywords))  # 去重

    # ============ 辅助方法 ============

    def _empty_layer_result(
        self,
        layer_name: str,
        error: Optional[str] = None
    ) -> LayerExecutionResult:
        """创建空的层级结果"""
        return LayerExecutionResult(
            layer_name=layer_name,
            success=False,
            results=[],
            execution_time_ms=0.0,
            confidence=0.0,
            metadata={},
            error=error
        )

    def _build_final_result(
        self,
        query: str,
        domain: str,
        layer1: LayerExecutionResult,
        layer2: LayerExecutionResult,
        layer3: LayerExecutionResult,
        start_time: datetime
    ) -> ThreeLayerResult:
        """构建最终结果"""
        # 确定最终结果来源
        if layer3.success and layer3.results:
            final_results = layer3.results
            reasoning = f"三层检索完成: Layer1({len(layer1.results)}) → Layer2({len(layer2.results)}) → Layer3({len(layer3.results)}) 最终推荐"
        elif layer2.success and layer2.results:
            final_results = layer2.results[:10]
            reasoning = f"部分检索: Layer1({len(layer1.results)}) → Layer2({len(layer2.results)}) 图谱推荐"
        elif layer1.success and layer1.results:
            final_results = layer1.results[:10]
            reasoning = f"基础检索: Layer1({len(layer1.results)}) 向量推荐"
        else:
            final_results = []
            reasoning = "检索失败: 未找到任何结果"

        # 计算总置信度
        layer_confidences = [
            layer1.confidence * 0.3,
            layer2.confidence * 0.4,
            layer3.confidence * 0.3
        ]
        total_confidence = sum(layer_confidences)

        # 计算总耗时
        total_time = (datetime.now() - start_time).total_seconds() * 1000

        return ThreeLayerResult(
            query=query,
            domain=domain,
            final_results=final_results,
            layer_1_result=layer1,
            layer_2_result=layer2,
            layer_3_result=layer3,
            total_confidence=total_confidence,
            total_execution_time_ms=total_time,
            reasoning=reasoning,
            metadata={
                "neo4j_direct_used": layer2.metadata.get("source") == "neo4j_direct",
                "layer_execution_times": {
                    "layer1": layer1.execution_time_ms,
                    "layer2": layer2.execution_time_ms,
                    "layer3": layer3.execution_time_ms
                },
                "stats": self.get_stats()
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        stats = self.stats.copy()

        if stats["total_queries"] > 0:
            stats["success_rate"] = (stats["layer3_success"] / stats["total_queries"]) * 100
            stats["neo4j_direct_rate"] = (stats["layer2_neo4j_direct"] / stats["total_queries"]) * 100
            stats["api_fallback_rate"] = (stats["layer2_api_fallback"] / stats["total_queries"]) * 100
        else:
            stats["success_rate"] = 0
            stats["neo4j_direct_rate"] = 0
            stats["api_fallback_rate"] = 0

        stats["neo4j_available"] = self.neo4j_available

        return stats

    def close(self):
        """关闭引擎和所有连接"""
        if self.neo4j_manager:
            self.neo4j_manager.close()
        logger.info("三层检索引擎已关闭")


# ============ 模块导出 ============

__all__ = [
    "TrueThreeLayerEngine",
    "ThreeLayerResult",
    "LayerExecutionResult",
    "Neo4jConnectionManager"
]
