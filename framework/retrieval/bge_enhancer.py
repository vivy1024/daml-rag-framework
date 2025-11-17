# -*- coding: utf-8 -*-
"""
DAML-RAG BGE增强器 v2.0

实现BGE-M3多功能embedding增强：
- Dense向量表示
- Sparse稀疏向量
- ColBERT重排向量
- 智能策略推荐

版本：v2.0.0
更新日期：2025-11-17
设计原则：多模式向量、智能策略、性能优化
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BGEMode(Enum):
    """BGE-M3模式"""
    DENSE = "dense"           # 密集向量模式
    SPARSE = "sparse"         # 稀疏向量模式
    COLBERT = "colbert"      # ColBERT重排模式
    HYBRID = "hybrid"         # 混合模式


class RetrievalStrategy(Enum):
    """检索策略"""
    DENSE_FIRST = "dense_first"     # 优先使用dense
    SPARSE_FIRST = "sparse_first"   # 优先使用sparse
    COLBERT_RERANK = "colbert_rerank"  # ColBERT重排
    MULTI_STAGE = "multi_stage"    # 多阶段策略


@dataclass
class BGEConfig:
    """BGE配置"""
    # 模型配置
    model_name: str = "BAAI/bge-m3"
    model_path: Optional[str] = None
    device: str = "cpu"
    max_length: int = 512

    # 向量维度
    dense_dim: int = 1024
    sparse_vocab_size: int = 30522
    colbert_dim: int = 1024

    # 策略配置
    default_mode: BGEMode = BGEMode.DENSE
    enable_auto_strategy: bool = True
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'dense': 0.4,
        'sparse': 0.3,
        'colbert': 0.3
    })

    # 性能配置
    batch_size: int = 32
    normalize_embeddings: bool = True
    use_fp16: bool = False

    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 10000


@dataclass
class BGEResult:
    """BGE结果"""
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[int, float]] = None
    colbert_vector: Optional[List[List[float]]] = None
    mode_used: BGEMode = BGEMode.DENSE
    strategy_used: RetrievalStrategy = RetrievalStrategy.DENSE_FIRST
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BGEEnhancer:
    """
    BGE-M3增强器

    提供多模式embedding生成和智能策略推荐。
    """

    def __init__(self, name: str = "BGEEnhancer", version: str = "2.0.0"):
        self.name = name
        self.version = version
        self._config = BGEConfig()
        self._model = None
        self._tokenizer = None
        self._cache = {}

        # 性能指标
        self._metrics = {
            'total_encodings': 0,
            'dense_encodings': 0,
            'sparse_encodings': 0,
            'colbert_encodings': 0,
            'average_encoding_time': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in RetrievalStrategy},
            'cache_hits': 0
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化BGE增强器"""
        try:
            if config:
                self._update_config(config)

            # 加载模型
            await self._load_model()

            logger.info(f"✅ BGE增强器初始化成功: {self.name}")
            return True

        except Exception as e:
            logger.error(f"❌ BGE增强器初始化失败 {self.name}: {e}")
            return False

    async def _load_model(self) -> None:
        """加载BGE-M3模型"""
        try:
            # 这里应该实际加载BGE-M3模型
            # 示例实现：
            # from FlagEmbedding import BGEM3FlagModel
            # self._model = BGEM3FlagModel(
            #     self._config.model_name,
            #     device=self._config.device,
            #     use_fp16=self._config.use_fp16
            # )

            # 临时模拟模型加载
            logger.info(f"加载BGE-M3模型: {self._config.model_name}")
            self._model = "mock_bge_model"  # 模拟模型对象

        except Exception as e:
            logger.error(f"加载BGE-M3模型失败: {e}")
            raise

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        if 'model_name' in config:
            self._config.model_name = config['model_name']
        if 'device' in config:
            self._config.device = config['device']
        if 'default_mode' in config:
            self._config.default_mode = BGEMode(config['default_mode'])
        if 'enable_auto_strategy' in config:
            self._config.enable_auto_strategy = config['enable_auto_strategy']

        logger.info(f"BGE配置已更新: {self.name}")

    async def encode(
        self,
        texts: Union[str, List[str]],
        mode: Optional[BGEMode] = None,
        strategy: Optional[RetrievalStrategy] = None
    ) -> Union[BGEResult, List[BGEResult]]:
        """编码文本"""
        start_time = asyncio.get_event_loop().time()

        # 统一输入格式
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts

        try:
            # 确定策略
            if strategy and self._config.enable_auto_strategy:
                strategy = strategy
            else:
                strategy = await self._recommend_strategy(texts_list[0])

            # 确定模式
            mode = mode or self._default_mode_for_strategy(strategy)

            # 检查缓存
            if self._config.enable_cache:
                cached_results = self._get_from_cache(texts_list, mode, strategy)
                if cached_results:
                    self._metrics['cache_hits'] += len(texts_list)
                    return cached_results[0] if is_single else cached_results

            # 编码文本
            results = []
            for text in texts_list:
                result = await self._encode_single(text, mode, strategy)
                results.append(result)

            # 缓存结果
            if self._config.enable_cache:
                self._cache_results(texts_list, results)

            # 更新指标
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(strategy, execution_time, len(texts_list))

            return results[0] if is_single else results

        except Exception as e:
            logger.error(f"BGE编码失败: {e}")
            # 返回空结果
            empty_result = BGEResult(
                mode_used=mode or BGEMode.DENSE,
                strategy_used=strategy or RetrievalStrategy.DENSE_FIRST,
                processing_time=asyncio.get_event_loop().time() - start_time,
                metadata={'error': str(e)}
            )
            return empty_result if is_single else [empty_result] * len(texts_list)

    async def _encode_single(
        self,
        text: str,
        mode: BGEMode,
        strategy: RetrievalStrategy
    ) -> BGEResult:
        """编码单个文本"""
        result = BGEResult(mode_used=mode, strategy_used=strategy)

        try:
            if mode == BGEMode.DENSE:
                result.dense_vector = await self._encode_dense(text)
                self._metrics['dense_encodings'] += 1
            elif mode == BGEMode.SPARSE:
                result.sparse_vector = await self._encode_sparse(text)
                self._metrics['sparse_encodings'] += 1
            elif mode == BGEMode.COLBERT:
                result.colbert_vector = await self._encode_colbert(text)
                self._metrics['colbert_encodings'] += 1
            elif mode == BGEMode.HYBRID:
                # 混合模式：生成所有类型的向量
                result.dense_vector = await self._encode_dense(text)
                result.sparse_vector = await self._encode_sparse(text)
                result.colbert_vector = await self._encode_colbert(text)
                self._metrics['dense_encodings'] += 1
                self._metrics['sparse_encodings'] += 1
                self._metrics['colbert_encodings'] += 1

        except Exception as e:
            logger.error(f"编码文本失败: {e}")
            result.metadata['encoding_error'] = str(e)

        return result

    async def _encode_dense(self, text: str) -> List[float]:
        """生成密集向量"""
        # 实际实现应该调用BGE-M3模型的dense编码功能
        # 示例实现：
        # dense_embeddings = self._model.encode([text], mode='dense')['dense_vecs']
        # return dense_embeddings[0].tolist()

        # 临时实现：生成随机向量
        return np.random.rand(self._config.dense_dim).tolist()

    async def _encode_sparse(self, text: str) -> Dict[int, float]:
        """生成稀疏向量"""
        # 实际实现应该调用BGE-M3模型的sparse编码功能
        # 示例实现：
        # sparse_embeddings = self._model.encode([text], mode='sparse')['lexical_weights']
        # return {k: float(v) for k, v in sparse_embeddings[0].items()}

        # 临时实现：生成随机稀疏向量
        sparse_vector = {}
        num_terms = min(len(text.split()), 100)  # 限制稀疏向量大小
        for i in range(num_terms):
            token_id = np.random.randint(0, self._config.sparse_vocab_size)
            weight = np.random.random()
            sparse_vector[token_id] = weight

        return sparse_vector

    async def _encode_colbert(self, text: str) -> List[List[float]]:
        """生成ColBERT向量"""
        # 实际实现应该调用BGE-M3模型的colbert编码功能
        # 示例实现：
        # colbert_embeddings = self._model.encode([text], mode='colbert')['colbert_vecs']
        # return colbert_embeddings[0].tolist()

        # 临时实现：生成随机ColBERT向量
        num_tokens = min(len(text.split()), 128)  # 限制ColBERT向量长度
        return [np.random.rand(self._config.colbert_dim).tolist() for _ in range(num_tokens)]

    async def recommend_strategy(self, query: str) -> RetrievalStrategy:
        """推荐检索策略"""
        return await self._recommend_strategy(query)

    async def _recommend_strategy(self, query: str) -> RetrievalStrategy:
        """智能策略推荐"""
        query_length = len(query.split())
        query_words = query.lower().split()

        # 短查询：使用dense模式
        if query_length <= 3:
            return RetrievalStrategy.DENSE_FIRST

        # 包含专业术语：使用sparse模式
        domain_keywords = ['深蹲', '卧推', '硬拉', '引体向上', '俯卧撑', '平板支撑']
        if any(keyword in query for keyword in domain_keywords):
            return RetrievalStrategy.SPARSE_FIRST

        # 长查询：使用ColBERT重排
        if query_length > 10:
            return RetrievalStrategy.COLBERT_RERANK

        # 包含比较词：使用多阶段策略
        comparison_words = ['对比', '比较', '区别', '差异', '哪个好']
        if any(word in query for word in comparison_words):
            return RetrievalStrategy.MULTI_STAGE

        # 默认使用dense优先
        return RetrievalStrategy.DENSE_FIRST

    def _default_mode_for_strategy(self, strategy: RetrievalStrategy) -> BGEMode:
        """获取策略的默认模式"""
        strategy_mode_map = {
            RetrievalStrategy.DENSE_FIRST: BGEMode.DENSE,
            RetrievalStrategy.SPARSE_FIRST: BGEMode.SPARSE,
            RetrievalStrategy.COLBERT_RERANK: BGEMode.COLBERT,
            RetrievalStrategy.MULTI_STAGE: BGEMode.HYBRID
        }
        return strategy_mode_map.get(strategy, BGEMode.DENSE)

    async def compute_similarity(
        self,
        query_result: BGEResult,
        doc_result: BGEResult,
        mode: Optional[BGEMode] = None
    ) -> float:
        """计算相似度"""
        if mode == BGEMode.DENSE or (not mode and query_result.dense_vector and doc_result.dense_vector):
            return self._cosine_similarity(query_result.dense_vector, doc_result.dense_vector)
        elif mode == BGEMode.SPARSE or (not mode and query_result.sparse_vector and doc_result.sparse_vector):
            return self._sparse_similarity(query_result.sparse_vector, doc_result.sparse_vector)
        elif mode == BGEMode.COLBERT or (not mode and query_result.colbert_vector and doc_result.colbert_vector):
            return self._colbert_similarity(query_result.colbert_vector, doc_result.colbert_vector)
        elif mode == BGEMode.HYBRID:
            # 混合模式：加权平均
            similarities = []
            weights = self._config.strategy_weights

            if query_result.dense_vector and doc_result.dense_vector:
                similarities.append(
                    weights['dense'] * self._cosine_similarity(query_result.dense_vector, doc_result.dense_vector)
                )
            if query_result.sparse_vector and doc_result.sparse_vector:
                similarities.append(
                    weights['sparse'] * self._sparse_similarity(query_result.sparse_vector, doc_result.sparse_vector)
                )
            if query_result.colbert_vector and doc_result.colbert_vector:
                similarities.append(
                    weights['colbert'] * self._colbert_similarity(query_result.colbert_vector, doc_result.colbert_vector)
                )

            return sum(similarities) if similarities else 0.0
        else:
            return 0.0

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        # 归一化
        if self._config.normalize_embeddings:
            vec1_norm = vec1_np / np.linalg.norm(vec1_np)
            vec2_norm = vec2_np / np.linalg.norm(vec2_np)
        else:
            vec1_norm = vec1_np
            vec2_norm = vec2_np

        # 计算余弦相似度
        dot_product = np.dot(vec1_norm, vec2_norm)
        return float(dot_product)

    def _sparse_similarity(self, sparse1: Dict[int, float], sparse2: Dict[int, float]) -> float:
        """计算稀疏向量相似度"""
        if not sparse1 or not sparse2:
            return 0.0

        # 计算交集
        intersection = set(sparse1.keys()) & set(sparse2.keys())
        if not intersection:
            return 0.0

        # 计算点积
        dot_product = sum(sparse1[i] * sparse2[i] for i in intersection)

        # 计算范数
        norm1 = np.sqrt(sum(weight ** 2 for weight in sparse1.values()))
        norm2 = np.sqrt(sum(weight ** 2 for weight in sparse2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _colbert_similarity(self, colbert1: List[List[float]], colbert2: List[List[float]]) -> float:
        """计算ColBERT相似度（MaxSim）"""
        if not colbert1 or not colbert2:
            return 0.0

        # MaxSim: 对于每个查询向量，找到最相似的文档向量
        max_similarities = []
        for q_vec in colbert1:
            q_vec_np = np.array(q_vec)
            similarities = []

            for d_vec in colbert2:
                d_vec_np = np.array(d_vec)
                # 计算余弦相似度
                q_norm = q_vec_np / np.linalg.norm(q_vec_np)
                d_norm = d_vec_np / np.linalg.norm(d_vec_np)
                similarity = np.dot(q_norm, d_norm)
                similarities.append(similarity)

            if similarities:
                max_similarities.append(max(similarities))

        # 返回平均最大相似度
        return np.mean(max_similarities) if max_similarities else 0.0

    def _get_from_cache(
        self,
        texts: List[str],
        mode: BGEMode,
        strategy: RetrievalStrategy
    ) -> Optional[List[BGEResult]]:
        """从缓存获取结果"""
        cached_results = []
        for text in texts:
            cache_key = self._generate_cache_key(text, mode, strategy)
            if cache_key in self._cache:
                cached_results.append(self._cache[cache_key])
            else:
                return None  # 部分缓存未命中，返回None

        return cached_results

    def _cache_results(self, texts: List[str], results: List[BGEResult]) -> None:
        """缓存结果"""
        for text, result in zip(texts, results):
            cache_key = self._generate_cache_key(
                text, result.mode_used, result.strategy_used
            )
            self._cache[cache_key] = result

        # 清理缓存（如果超过大小限制）
        if len(self._cache) > self._config.cache_size:
            # 简单的LRU：删除一半缓存
            items_to_remove = len(self._cache) // 2
            keys_to_remove = list(self._cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._cache[key]

    def _generate_cache_key(self, text: str, mode: BGEMode, strategy: RetrievalStrategy) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{text}_{mode.value}_{strategy.value}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_metrics(self, strategy: RetrievalStrategy, execution_time: float, batch_size: int) -> None:
        """更新指标"""
        # 更新策略使用统计
        self._metrics['strategy_usage'][strategy.value] += 1

        # 更新编码统计
        self._metrics['total_encodings'] += batch_size

        # 更新平均时间
        total_encodings = self._metrics['total_encodings']
        current_avg = self._metrics['average_encoding_time']
        self._metrics['average_encoding_time'] = (
            (current_avg * (total_encodings - batch_size) + execution_time) / total_encodings
        )

    async def batch_encode(
        self,
        texts: List[str],
        mode: Optional[BGEMode] = None,
        batch_size: Optional[int] = None
    ) -> List[BGEResult]:
        """批量编码"""
        batch_size = batch_size or self._config.batch_size
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await self.encode(batch, mode=mode)
            if isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)

        return all_results

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self._config.model_name,
            'device': self._config.device,
            'dense_dimension': self._config.dense_dim,
            'sparse_vocab_size': self._config.sparse_vocab_size,
            'colbert_dimension': self._config.colbert_dim,
            'max_length': self._config.max_length,
            'is_loaded': self._model is not None
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取BGE增强器指标"""
        return {
            'total_encodings': self._metrics['total_encodings'],
            'dense_encodings': self._metrics['dense_encodings'],
            'sparse_encodings': self._metrics['sparse_encodings'],
            'colbert_encodings': self._metrics['colbert_encodings'],
            'average_encoding_time': self._metrics['average_encoding_time'],
            'cache_hits': self._metrics['cache_hits'],
            'cache_hit_rate': (
                self._metrics['cache_hits'] / max(self._metrics['total_encodings'], 1)
            ),
            'cache_size': len(self._cache),
            'strategy_usage': self._metrics['strategy_usage']
        }


# 导出
__all__ = [
    'BGEEnhancer',
    'BGEConfig',
    'BGEResult',
    'BGEMode',
    'RetrievalStrategy'
]