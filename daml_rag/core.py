"""
DAML-RAG健身框架核心实现
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import time

from .config import DAMLRAGConfig
from .interfaces import (
    IRetriever,
    IOrchestrator,
    IModelProvider,
    IMemoryManager,
    ITaskScheduler,
    ITaskExecutor,
)
from .models import (
    QueryResult,
    RetrievalResult,
    Experience,
    Task,
    Workflow,
    TaskResult,
    TaskStatus,
)
from .base import ComponentRegistry
from .utils import setup_logging

# 导入三层检索系统
from ..daml_rag_retrieval.three_tier import ThreeTierRetriever, RetrievalRequest, ThreeTierRetrievalResult
from ..daml_rag_retrieval.vector.qdrant import QdrantVectorRetriever, QdrantConfig
from ..daml_rag_retrieval.vector.faiss import FAISSVectorRetriever, FAISSConfig
from ..daml_rag_retrieval.knowledge.neo4j import Neo4jKnowledgeRetriever, Neo4jConfig
from ..daml_rag_retrieval.rules.engine import RuleEngine, RuleContext


class DAMLRAGFramework:
    """玉珍健身框架主类"""

    def __init__(self, config: DAMLRAGConfig):
        self.config = config
        self.logger = setup_logging(self.config.logging, __name__)

        # 组件注册表
        self.registry = ComponentRegistry()

        # 核心组件
        self.retriever: Optional[IRetriever] = None
        self.orchestrator: Optional[IOrchestrator] = None
        self.model_provider: Optional[IModelProvider] = None
        self.memory_manager: Optional[IMemoryManager] = None
        self.task_scheduler: Optional[ITaskScheduler] = None
        self.task_executor: Optional[ITaskExecutor] = None

        # 三层检索系统组件
        self.three_tier_retriever: Optional[ThreeTierRetriever] = None
        self.vector_retriever: Optional[QdrantVectorRetriever] = None
        self.knowledge_retriever: Optional[Neo4jKnowledgeRetriever] = None
        self.rule_engine: Optional[RuleEngine] = None

        # 运行时状态
        self._initialized = False
        self._shutdown = False
        self._active_queries: Set[str] = set()
        self._query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    async def initialize(self) -> None:
        """初始化框架"""
        if self._initialized:
            self.logger.warning("框架已经初始化")
            return

        try:
            self.logger.info("开始初始化玉珍健身框架...")

            # 验证配置
            config_errors = self.config.validate()
            if config_errors:
                raise ValueError(f"配置验证失败: {', '.join(config_errors)}")

            # 初始化核心组件
            await self._setup_components()

            # 启动后台任务
            await self._start_background_tasks()

            self._initialized = True
            self.logger.info("玉珍健身框架初始化完成")

        except Exception as e:
            self.logger.error(f"框架初始化失败: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """关闭框架"""
        if self._shutdown:
            return

        self.logger.info("开始关闭玉珍健身框架...")
        self._shutdown = True

        # 等待活跃查询完成
        if self._active_queries:
            self.logger.info(f"等待 {len(self._active_queries)} 个活跃查询完成...")
            for _ in range(30):  # 最多等待30秒
                if not self._active_queries:
                    break
                await asyncio.sleep(1)

        # 停止后台任务
        await self._stop_background_tasks()

        # 清理组件
        await self._cleanup_components()

        self._initialized = False
        self.logger.info("玉珍健身框架已关闭")

    async def process_query(self, query: str,
                          context: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None) -> QueryResult:
        """处理用户查询"""
        if not self._initialized:
            raise RuntimeError("框架未初始化")

        if self._shutdown:
            raise RuntimeError("框架已关闭")

        # 生成查询ID
        query_id = f"{int(time.time())}-{hash(query) % 10000}"
        self._active_queries.add(query_id)

        try:
            start_time = time.time()
            self.logger.info(f"处理查询 [{query_id}]: {query[:100]}...")

            context = context or {}
            context['user_id'] = user_id
            context['query_id'] = query_id

            # 更新统计
            self._query_stats['total_queries'] += 1

            # 1. 预处理查询
            processed_query, context = await self._preprocess_query(query, context)

            # 2. 检索相关经验
            experiences = await self._retrieve_similar_experiences(processed_query, context)

            # 3. 决定处理策略
            strategy = await self._determine_processing_strategy(processed_query, experiences, context)

            # 4. 执行处理
            if strategy['use_orchestrator']:
                result = await self._orchestrated_process(processed_query, experiences, context)
            else:
                result = await self._simple_process(processed_query, experiences, context)

            # 5. 后处理结果
            result = await self._postprocess_result(result, context)

            # 6. 存储经验
            await self._store_experience(result, context)

            # 7. 更新统计
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            self._update_query_stats(result, execution_time)

            self.logger.info(f"查询 [{query_id}] 处理完成，耗时 {execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"查询 [{query_id}] 处理失败: {str(e)}")
            self._query_stats['failed_queries'] += 1

            # 返回错误结果
            error_result = QueryResult(
                query=query,
                response="抱歉，处理您的查询时遇到了问题，请稍后重试。",
                model_used="error",
                execution_time=0.0,
                context=context or {},
                metadata={'error': str(e)}
            )

            return error_result

        finally:
            self._active_queries.discard(query_id)

    async def _setup_components(self) -> None:
        """设置组件"""
        self.logger.info("设置核心组件...")

        # 从注册表获取组件
        self.retriever = self.registry.get_component(IRetriever)
        self.orchestrator = self.registry.get_component(IOrchestrator)
        self.model_provider = self.registry.get_component(IModelProvider)
        self.memory_manager = self.registry.get_component(IMemoryManager)
        self.task_scheduler = self.registry.get_component(ITaskScheduler)
        self.task_executor = self.registry.get_component(ITaskExecutor)

        # 初始化三层检索系统
        await self._setup_three_tier_retrieval()

        # 验证必需组件
        required_components = [
            (self.retriever, "IRetriever"),
            (self.model_provider, "IModelProvider"),
            (self.memory_manager, "IMemoryManager"),
        ]

        for component, name in required_components:
            if component is None:
                raise ValueError(f"缺少必需组件: {name}")

        self.logger.info("核心组件设置完成")

    async def _setup_three_tier_retrieval(self) -> None:
        """设置三层检索系统"""
        self.logger.info("初始化三层检索系统...")

        try:
            # 1. 初始化向量检索器 (默认使用Qdrant)
            vector_config = getattr(self.config, 'vector_config', None)
            if vector_config:
                if isinstance(vector_config, QdrantConfig):
                    self.vector_retriever = QdrantVectorRetriever(vector_config)
                elif isinstance(vector_config, FAISSConfig):
                    self.vector_retriever = FAISSVectorRetriever(vector_config)
                else:
                    # 创建默认Qdrant配置
                    default_qdrant_config = QdrantConfig()
                    self.vector_retriever = QdrantVectorRetriever(default_qdrant_config)
            else:
                # 使用默认配置
                default_qdrant_config = QdrantConfig()
                self.vector_retriever = QdrantVectorRetriever(default_qdrant_config)

            await self.vector_retriever.initialize()

            # 2. 初始化知识图谱检索器
            knowledge_config = getattr(self.config, 'knowledge_config', None)
            if knowledge_config and isinstance(knowledge_config, Neo4jConfig):
                self.knowledge_retriever = Neo4jKnowledgeRetriever(knowledge_config)
            else:
                # 使用默认配置
                default_neo4j_config = Neo4jConfig()
                self.knowledge_retriever = Neo4jKnowledgeRetriever(default_neo4j_config)

            await self.knowledge_retriever.initialize()

            # 3. 初始化规则引擎
            self.rule_engine = RuleEngine()

            # 4. 创建三层检索器
            self.three_tier_retriever = ThreeTierRetriever(
                vector_retriever=self.vector_retriever,
                knowledge_retriever=self.knowledge_retriever,
                rule_engine=self.rule_engine,
                weights={
                    'vector': getattr(self.config, 'vector_weight', 0.4),
                    'knowledge': getattr(self.config, 'knowledge_weight', 0.4),
                    'rules': getattr(self.config, 'rules_weight', 0.2)
                },
                cache_enabled=getattr(self.config, 'cache_enabled', True),
                cache_ttl=getattr(self.config, 'cache_ttl', 300)
            )

            await self.three_tier_retriever.initialize()

            self.logger.info("三层检索系统初始化完成")

        except Exception as e:
            self.logger.error(f"三层检索系统初始化失败: {str(e)}")
            # 不抛出异常，允许系统继续使用传统检索器
            self.logger.warning("回退到传统检索模式")

    async def _preprocess_query(self, query: str, context: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """预处理查询"""
        # 基本清理
        processed_query = query.strip()

        # 可以在这里添加更多的预处理逻辑
        # - 查询扩展
        # - 意图识别
        # - 实体提取

        return processed_query, context

    async def _retrieve_similar_experiences(self, query: str,
                                          context: Dict[str, Any]) -> List[Experience]:
        """检索相似经验"""
        try:
            experiences = await self.memory_manager.retrieve_similar_experiences(
                query=query,
                top_k=self.config.learning.max_experiences_per_query,
                filters=context.get('filters')
            )
            self.logger.debug(f"检索到 {len(experiences)} 个相关经验")
            return experiences
        except Exception as e:
            self.logger.warning(f"经验检索失败: {str(e)}")
            return []

    async def _determine_processing_strategy(self, query: str,
                                           experiences: List[Experience],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """决定处理策略"""
        strategy = {
            'use_orchestrator': False,
            'model_type': 'student',
            'complexity_score': 0.0,
            'reason': ''
        }

        # 计算查询复杂度
        complexity_score = await self._calculate_query_complexity(query, context)
        strategy['complexity_score'] = complexity_score

        # 简单查询检测
        if len(query.strip()) <= 5:
            strategy['reason'] = '简单查询'
            return strategy

        # 复杂关键词检测
        complex_keywords = ['计划', '方案', '设计', '推荐', '分析', '评估', '制定']
        if any(keyword in query for keyword in complex_keywords):
            strategy['use_orchestrator'] = True
            strategy['reason'] = '包含复杂关键词'
            return strategy

        # 经验不足检测
        if len(experiences) < 2:
            strategy['use_orchestrator'] = True
            strategy['reason'] = '经验不足'
            return strategy

        # 复杂度阈值检测
        if complexity_score > self.config.learning.adaptive_threshold:
            strategy['use_orchestrator'] = True
            strategy['model_type'] = 'teacher'
            strategy['reason'] = '高复杂度查询'
            return strategy

        strategy['reason'] = '标准处理'
        return strategy

    async def _calculate_query_complexity(self, query: str,
                                         context: Dict[str, Any]) -> float:
        """计算查询复杂度"""
        complexity = 0.0

        # 长度复杂度
        length_score = min(len(query) / 100.0, 1.0)
        complexity += length_score * 0.2

        # 关键词复杂度
        complex_keywords = ['计划', '方案', '设计', '推荐', '分析', '评估', '制定', '康复', '营养']
        keyword_count = sum(1 for keyword in complex_keywords if keyword in query)
        keyword_score = min(keyword_count / 3.0, 1.0)
        complexity += keyword_score * 0.4

        # 问号复杂度（问题数量）
        question_count = query.count('？') + query.count('?')
        question_score = min(question_count / 2.0, 1.0)
        complexity += question_score * 0.2

        # 条件复杂度（如果、那么等）
        condition_keywords = ['如果', '那么', '否则', '但是', '然而', '同时']
        condition_count = sum(1 for keyword in condition_keywords if keyword in query)
        condition_score = min(condition_count / 2.0, 1.0)
        complexity += condition_score * 0.2

        return min(complexity, 1.0)

    async def _orchestrated_process(self, query: str,
                                  experiences: List[Experience],
                                  context: Dict[str, Any]) -> QueryResult:
        """编排器处理"""
        try:
            self.logger.debug("使用编排器处理查询")

            # 构建任务工作流
            workflow = await self._build_workflow(query, experiences, context)

            # 执行工作流
            execution_result = await self.orchestrator.execute_workflow(workflow, context)

            # 生成最终响应
            result = await self._generate_response(query, execution_result, experiences, context)

            # 设置来源
            if execution_result:
                result.sources.extend(execution_result.keys())

            return result

        except Exception as e:
            self.logger.warning(f"编排器处理失败，降级到简单处理: {str(e)}")
            return await self._simple_process(query, experiences, context)

    async def _simple_process(self, query: str,
                            experiences: List[Experience],
                            context: Dict[str, Any]) -> QueryResult:
        """简单处理"""
        try:
            self.logger.debug("使用简单处理模式")

            # 使用三层检索系统 (如果可用) 或传统检索器
            if self.three_tier_retriever:
                retrieval_result = await self._three_tier_retrieve(query, context)
            else:
                # 回退到传统检索器
                retrieval_result = await self.retriever.retrieve(
                    query=query,
                    top_k=self.config.retrieval.top_k,
                    filters=context.get('filters')
                )

            # 选择模型
            model_provider = await self._select_model(query, retrieval_result, context)

            # 构建提示
            prompt = await self._build_prompt(query, retrieval_result, experiences, context)

            # 生成响应
            generation_result = await model_provider.generate(prompt, context)

            # 创建结果
            result = QueryResult(
                query=query,
                response=generation_result.content,
                sources=retrieval_result.sources,
                model_used=model_provider.get_model_info().get('name', 'unknown'),
                execution_time=0.0,  # 将在上层设置
                context=context,
                metadata={
                    'retrieval_result': retrieval_result,
                    'generation_result': generation_result,
                    'experiences_used': len(experiences),
                    'retrieval_method': 'three_tier' if self.three_tier_retriever else 'traditional'
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"简单处理失败: {str(e)}")
            raise

    async def _three_tier_retrieve(self, query: str, context: Dict[str, Any]) -> RetrievalResult:
        """使用三层检索系统进行检索"""
        try:
            # 构建检索请求
            retrieval_request = RetrievalRequest(
                query=query,
                top_k=getattr(self.config, 'top_k', 10),
                score_threshold=getattr(self.config, 'score_threshold', 0.0),
                filters=context.get('filters'),
                include_metadata=True,
                user_id=context.get('user_id'),
                session_id=context.get('session_id')
            )

            # 执行三层检索
            three_tier_result = await self.three_tier_retriever.retrieve(retrieval_request)

            # 转换为标准RetrievalResult格式
            retrieval_result = RetrievalResult(
                query=query,
                documents=[],
                sources=[],
                scores=[],
                total_found=0,
                retrieval_metadata={
                    'three_tier_result': three_tier_result,
                    'vector_count': len(three_tier_result.vector_results.documents) if three_tier_result.vector_results else 0,
                    'knowledge_count': len(three_tier_result.knowledge_results.documents) if three_tier_result.knowledge_results else 0,
                    'rules_count': len(three_tier_result.final_results.documents) if three_tier_result.final_results else 0,
                    'execution_time': three_tier_result.total_execution_time
                }
            )

            # 处理最终结果
            if three_tier_result.final_results and three_tier_result.final_results.documents:
                retrieval_result.documents = three_tier_result.final_results.documents
                retrieval_result.sources = three_tier_result.final_results.sources or []
                retrieval_result.scores = three_tier_result.final_results.scores or []
                retrieval_result.total_found = three_tier_result.final_results.total_found

            self.logger.debug(f"三层检索完成: 向量={retrieval_result.retrieval_metadata['vector_count']}, "
                            f"知识={retrieval_result.retrieval_metadata['knowledge_count']}, "
                            f"最终={retrieval_result.total_found}")

            return retrieval_result

        except Exception as e:
            self.logger.error(f"三层检索失败: {str(e)}")
            # 回退到传统检索
            return await self.retriever.retrieve(
                query=query,
                top_k=getattr(self.config, 'top_k', 10),
                filters=context.get('filters')
            )

    async def _build_workflow(self, query: str,
                            experiences: List[Experience],
                            context: Dict[str, Any]) -> Workflow:
        """构建工作流"""
        # 这里应该根据具体领域和查询内容构建任务工作流
        # 暂时返回空工作流
        workflow = Workflow(
            name=f"query_workflow_{int(time.time())}",
            description=f"处理查询: {query[:50]}...",
            context=context
        )

        return workflow

    async def _select_model(self, query: str,
                          retrieval_result: RetrievalResult,
                          context: Dict[str, Any]) -> IModelProvider:
        """选择模型"""
        # 简单的模型选择逻辑
        # 在实际实现中，这里应该使用更复杂的策略

        # 检查是否有教师模型可用
        if hasattr(self, 'teacher_model') and self.teacher_model:
            # 复杂查询使用教师模型
            complexity = await self._calculate_query_complexity(query, context)
            if complexity > self.config.learning.adaptive_threshold:
                return self.teacher_model

        # 默认使用学生模型
        return self.model_provider

    async def _build_prompt(self, query: str,
                          retrieval_result: RetrievalResult,
                          experiences: List[Experience],
                          context: Dict[str, Any]) -> str:
        """构建提示"""
        prompt_parts = []

        # 系统提示
        system_prompt = await self._build_system_prompt(context)
        prompt_parts.append(system_prompt)

        # 相关知识
        if retrieval_result.documents:
            prompt_parts.append("\n相关知识：")
            for i, doc in enumerate(retrieval_result.documents[:3], 1):
                prompt_parts.append(f"{i}. {doc.content[:200]}...")

        # 经验示例
        if experiences:
            prompt_parts.append("\n相关经验：")
            for i, exp in enumerate(experiences[:2], 1):
                prompt_parts.append(f"经验{i}: Q: {exp.query}\nA: {exp.response[:100]}...")

        # 用户查询
        prompt_parts.append(f"\n用户问题：{query}")

        # 构建完整提示
        full_prompt = "\n".join(prompt_parts)

        # 检查提示长度并优化
        if hasattr(self.model_provider, 'estimate_tokens'):
            estimated_tokens = self.model_provider.estimate_tokens(full_prompt)
            max_tokens = getattr(self.config, 'max_prompt_tokens', 4000)

            if estimated_tokens > max_tokens:
                # 截断提示
                full_prompt = await self._optimize_prompt_length(full_prompt, max_tokens)

        return full_prompt

    async def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """构建系统提示"""
        domain = self.config.domain

        system_prompts = {
            'fitness': "你是一个专业的健身教练，请根据用户的问题提供专业、实用的健身建议。",
            'healthcare': "你是一个医疗健康顾问，请提供准确、负责的健康信息。",
            'education': "你是一个教育专家，请提供清晰、有价值的教育指导。",
            'default': "你是一个专业的AI助手，请根据用户提供的信息给出准确、有帮助的回答。"
        }

        return system_prompts.get(domain, system_prompts['default'])

    async def _optimize_prompt_length(self, prompt: str, max_tokens: int) -> str:
        """优化提示长度"""
        # 简单的截断策略
        # 在实际实现中应该使用更智能的方法
        lines = prompt.split('\n')
        optimized_lines = []
        current_length = 0

        for line in lines:
            line_length = len(line)
            if current_length + line_length > max_tokens * 0.8:  # 留一些余量
                break
            optimized_lines.append(line)
            current_length += line_length

        return '\n'.join(optimized_lines)

    async def _generate_response(self, query: str,
                               execution_result: Dict[str, Any],
                               experiences: List[Experience],
                               context: Dict[str, Any]) -> QueryResult:
        """生成响应"""
        # 这里应该根据工作流执行结果生成最终响应
        # 暂时使用简单的模板
        if execution_result:
            # 整合工作流结果
            result_summary = "\n".join([f"- {key}: {value}" for key, value in execution_result.items()])
            response = f"基于分析结果，为您提供以下建议：\n{result_summary}"
        else:
            response = "抱歉，当前无法为您生成具体的建议。"

        return QueryResult(
            query=query,
            response=response,
            sources=list(execution_result.keys()) if execution_result else [],
            model_used="orchestrator",
            execution_time=0.0,
            context=context,
            metadata={'workflow_results': execution_result}
        )

    async def _postprocess_result(self, result: QueryResult,
                                context: Dict[str, Any]) -> QueryResult:
        """后处理结果"""
        # 可以在这里添加结果后处理逻辑
        # - 格式化响应
        # - 添加引用
        # - 质量检查

        return result

    async def _store_experience(self, result: QueryResult,
                              context: Dict[str, Any]) -> None:
        """存储经验"""
        try:
            experience = result.to_experience()
            await self.memory_manager.store_experience(experience)
            self.logger.debug("经验存储成功")
        except Exception as e:
            self.logger.warning(f"经验存储失败: {str(e)}")

    async def _start_background_tasks(self) -> None:
        """启动后台任务"""
        if self.config.enable_performance_monitoring:
            # 启动性能监控任务
            asyncio.create_task(self._performance_monitoring_loop())

        # 启动健康检查任务
        asyncio.create_task(self._health_check_loop())

    async def _stop_background_tasks(self) -> None:
        """停止后台任务"""
        # 后台任务会在shutdown标志设置时自动停止
        pass

    async def _cleanup_components(self) -> None:
        """清理组件"""
        # 清理组件资源
        if self.memory_manager:
            try:
                await self.memory_manager.cleanup_old_experiences()
            except Exception as e:
                self.logger.warning(f"清理经验失败: {str(e)}")

        # 清理三层检索系统
        if self.three_tier_retriever:
            try:
                await self.three_tier_retriever.close()
            except Exception as e:
                self.logger.warning(f"关闭三层检索器失败: {str(e)}")

        if self.vector_retriever:
            try:
                await self.vector_retriever.close()
            except Exception as e:
                self.logger.warning(f"关闭向量检索器失败: {str(e)}")

        if self.knowledge_retriever:
            try:
                await self.knowledge_retriever.close()
            except Exception as e:
                self.logger.warning(f"关闭知识图谱检索器失败: {str(e)}")

    async def _performance_monitoring_loop(self) -> None:
        """性能监控循环"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._shutdown:
                    break

                # 记录性能指标
                self.logger.debug(f"查询统计: {self._query_stats}")

            except Exception as e:
                self.logger.error(f"性能监控错误: {str(e)}")

    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._shutdown:
                    break

                # 检查组件健康状态
                await self._check_components_health()

            except Exception as e:
                self.logger.error(f"健康检查错误: {str(e)}")

    async def _check_components_health(self) -> None:
        """检查组件健康状态"""
        components_to_check = [
            (self.retriever, "retriever"),
            (self.model_provider, "model_provider"),
            (self.memory_manager, "memory_manager"),
            # 检查三层检索系统组件
            (self.three_tier_retriever, "three_tier_retriever"),
            (self.vector_retriever, "vector_retriever"),
            (self.knowledge_retriever, "knowledge_retriever"),
        ]

        for component, name in components_to_check:
            if component and hasattr(component, 'health_check'):
                try:
                    is_healthy = await component.health_check()
                    if not is_healthy:
                        self.logger.warning(f"组件 {name} 健康检查失败")
                except Exception as e:
                    self.logger.error(f"组件 {name} 健康检查异常: {str(e)}")

    def _update_query_stats(self, result: QueryResult, execution_time: float) -> None:
        """更新查询统计"""
        self._query_stats['successful_queries'] += 1

        # 更新平均响应时间
        total_successful = self._query_stats['successful_queries']
        current_avg = self._query_stats['average_response_time']
        new_avg = (current_avg * (total_successful - 1) + execution_time) / total_successful
        self._query_stats['average_response_time'] = new_avg

    def get_framework_stats(self) -> Dict[str, Any]:
        """获取框架统计信息"""
        stats = {
            'initialized': self._initialized,
            'shutdown': self._shutdown,
            'active_queries': len(self._active_queries),
            'query_stats': self._query_stats.copy(),
            'config_summary': {
                'domain': self.config.domain,
                'environment': self.config.environment,
                'debug': self.config.debug,
            },
            'three_tier_system': {
                'enabled': self.three_tier_retriever is not None,
                'vector_retriever': self.vector_retriever is not None,
                'knowledge_retriever': self.knowledge_retriever is not None,
                'rule_engine': self.rule_engine is not None,
            }
        }

        return stats

    async def get_detailed_framework_stats(self) -> Dict[str, Any]:
        """获取详细框架统计信息（包含异步组件统计）"""
        stats = self.get_framework_stats()

        # 添加三层检索系统统计
        if self.three_tier_retriever:
            try:
                three_tier_stats = await self.three_tier_retriever.get_statistics()
                stats['three_tier_system']['statistics'] = three_tier_stats
            except Exception as e:
                stats['three_tier_system']['statistics_error'] = str(e)

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """框架健康检查"""
        health_status = {
            'framework_healthy': self._initialized and not self._shutdown,
            'components': {},
            'overall_status': 'healthy'
        }

        # 检查各组件健康状态
        components_to_check = [
            (self.retriever, "retriever"),
            (self.orchestrator, "orchestrator"),
            (self.model_provider, "model_provider"),
            (self.memory_manager, "memory_manager"),
            # 检查三层检索系统组件
            (self.three_tier_retriever, "three_tier_retriever"),
            (self.vector_retriever, "vector_retriever"),
            (self.knowledge_retriever, "knowledge_retriever"),
        ]

        unhealthy_count = 0
        for component, name in components_to_check:
            if component is None:
                health_status['components'][name] = 'not_configured'
            elif hasattr(component, 'health_check'):
                try:
                    is_healthy = await component.health_check()
                    health_status['components'][name] = 'healthy' if is_healthy else 'unhealthy'
                    if not is_healthy:
                        unhealthy_count += 1
                except Exception as e:
                    health_status['components'][name] = f'error: {str(e)}'
                    unhealthy_count += 1
            else:
                health_status['components'][name] = 'unknown'

        # 规则引擎健康检查
        if self.rule_engine:
            try:
                # 规则引擎通常不会失败，这里做基本检查
                health_status['components']['rule_engine'] = 'healthy'
            except Exception as e:
                health_status['components']['rule_engine'] = f'error: {str(e)}'
                unhealthy_count += 1
        else:
            health_status['components']['rule_engine'] = 'not_configured'

        # 确定整体状态
        if unhealthy_count > 0:
            health_status['overall_status'] = 'degraded' if unhealthy_count < len(components_to_check) else 'unhealthy'

        return health_status