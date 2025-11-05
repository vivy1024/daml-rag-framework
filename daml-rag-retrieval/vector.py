"""
向量检索实现
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import logging

from daml_rag.interfaces import IVectorRetriever, IEmbeddingModel, ICacheManager
from daml_rag.models import Document, SimilarityResult, RetrievalResult, ValidationResult
from daml_rag.base import ConfigurableComponent


class VectorRetriever(IVectorRetriever, ConfigurableComponent):
    """向量检索器基类"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_model: Optional[IEmbeddingModel] = None
        self.cache_manager: Optional[ICacheManager] = None
        self.indexed_documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None

    async def _do_initialize(self) -> None:
        """初始化向量检索器"""
        # 初始化嵌入模型
        embedding_config = self.get_config_value('embedding_model', {})
        if embedding_config:
            self.embedding_model = self._create_embedding_model(embedding_config)
            if hasattr(self.embedding_model, 'initialize'):
                await self.embedding_model.initialize()

        # 初始化缓存管理器
        cache_config = self.get_config_value('cache', {})
        if cache_config.get('enabled', True):
            self.cache_manager = self._create_cache_manager(cache_config)
            if hasattr(self.cache_manager, 'initialize'):
                await self.cache_manager.initialize()

    def _create_embedding_model(self, config: Dict[str, Any]) -> IEmbeddingModel:
        """创建嵌入模型，子类实现"""
        raise NotImplementedError

    def _create_cache_manager(self, config: Dict[str, Any]) -> ICacheManager:
        """创建缓存管理器，子类实现"""
        raise NotImplementedError

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """文本向量化"""
        if not self.embedding_model:
            raise RuntimeError("嵌入模型未初始化")

        try:
            embeddings = await self.embedding_model.encode(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"文本向量化失败: {str(e)}")
            raise

    async def similarity_search(self, query_vector: List[float],
                              top_k: int = 5,
                              filters: Optional[Dict[str, Any]] = None) -> List[SimilarityResult]:
        """相似度搜索"""
        if self.document_embeddings is None:
            self.logger.warning("文档索引为空")
            return []

        try:
            # 转换为numpy数组
            query_np = np.array(query_vector).reshape(1, -1)

            # 计算相似度
            similarities = self._compute_similarities(query_np)

            # 应用过滤
            if filters:
                similarities = self._apply_filters(similarities, filters)

            # 获取top-k结果
            top_indices = np.argsort(similarities[0])[::-1][:top_k]

            results = []
            for idx in top_indices:
                if similarities[0][idx] > 0:  # 只返回有相似度的结果
                    doc = self.indexed_documents[idx]
                    result = SimilarityResult(
                        document=doc,
                        score=float(similarities[0][idx])
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"相似度搜索失败: {str(e)}")
            return []

    def _compute_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """计算相似度，子类实现"""
        raise NotImplementedError

    def _apply_filters(self, similarities: np.ndarray,
                      filters: Dict[str, Any]) -> np.ndarray:
        """应用过滤条件"""
        # 简单实现，子类可重写
        return similarities

    async def hybrid_search(self, query: str, query_vector: List[float],
                          top_k: int = 5, alpha: float = 0.7) -> List[SimilarityResult]:
        """混合搜索（文本+向量）"""
        try:
            # 向量搜索
            vector_results = await self.similarity_search(query_vector, top_k * 2)

            # 文本搜索（简单的关键词匹配）
            text_results = await self._text_search(query, top_k * 2)

            # 合并结果
            combined_results = self._combine_results(vector_results, text_results, alpha, top_k)
            return combined_results

        except Exception as e:
            self.logger.error(f"混合搜索失败: {str(e)}")
            return await self.similarity_search(query_vector, top_k)

    async def _text_search(self, query: str, top_k: int) -> List[SimilarityResult]:
        """文本搜索"""
        query_lower = query.lower()
        results = []

        for doc in self.indexed_documents:
            content_lower = doc.content.lower()

            # 简单的关键词匹配
            keyword_matches = sum(1 for word in query_lower.split() if word in content_lower)
            if keyword_matches > 0:
                score = keyword_matches / len(query_lower.split())
                results.append(SimilarityResult(document=doc, score=score))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _combine_results(self, vector_results: List[SimilarityResult],
                        text_results: List[SimilarityResult],
                        alpha: float, top_k: int) -> List[SimilarityResult]:
        """合并向量搜索和文本搜索结果"""
        # 创建文档ID到结果的映射
        doc_map = {}

        # 处理向量搜索结果
        for result in vector_results:
            doc_id = result.document.id
            doc_map[doc_id] = result
            doc_map[doc_id].score = result.score * alpha

        # 处理文本搜索结果
        for result in text_results:
            doc_id = result.document.id
            if doc_id in doc_map:
                doc_map[doc_id].score += result.score * (1 - alpha)
            else:
                result.score = result.score * (1 - alpha)
                doc_map[doc_id] = result

        # 排序并返回top-k
        combined_results = sorted(doc_map.values(), key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]

    async def retrieve(self, query: str, top_k: int = 5,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行检索"""
        import time
        start_time = time.time()

        try:
            # 检查缓存
            cache_key = f"retrieve:{hash(query)}:{top_k}"
            if self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    self.logger.debug("命中缓存")
                    return cached_result

            # 生成查询向量
            query_embeddings = await self.embed([query])
            query_vector = query_embeddings[0]

            # 执行相似度搜索
            similarity_results = await self.similarity_search(query_vector, top_k, filters)

            # 构建结果
            documents = [result.document for result in similarity_results]
            scores = [result.score for result in similarity_results]
            sources = [doc.id for doc in documents]

            result = RetrievalResult(
                query=query,
                documents=documents,
                scores=scores,
                metadata={
                    'retrieval_type': 'vector',
                    'similarity_threshold': self.get_config_value('similarity_threshold', 0.0),
                    'total_candidates': len(self.indexed_documents)
                },
                execution_time=time.time() - start_time
            )

            # 缓存结果
            if self.cache_manager:
                await self.cache_manager.set(cache_key, result, ttl=300)

            return result

        except Exception as e:
            self.logger.error(f"检索失败: {str(e)}")
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                metadata={'error': str(e)},
                execution_time=time.time() - start_time
            )

    async def batch_retrieve(self, queries: List[str],
                           top_k: int = 5) -> List[RetrievalResult]:
        """批量检索"""
        tasks = [self.retrieve(query, top_k) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"批量检索第{i}个查询失败: {str(result)}")
                processed_results.append(RetrievalResult(
                    query=queries[i],
                    documents=[],
                    scores=[],
                    metadata={'error': str(result)}
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def update_index(self, documents: List[Document]) -> bool:
        """更新索引"""
        try:
            # 检查是否有新文档
            new_docs = [doc for doc in documents if doc.id not in [d.id for d in self.indexed_documents]]

            if not new_docs:
                self.logger.debug("没有新文档需要索引")
                return True

            # 生成新文档的嵌入
            texts = [doc.content for doc in new_docs]
            embeddings = await self.embed(texts)

            # 更新文档列表和嵌入
            self.indexed_documents.extend(new_docs)

            if self.document_embeddings is None:
                self.document_embeddings = np.array(embeddings)
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, np.array(embeddings)])

            self.logger.info(f"成功索引 {len(new_docs)} 个新文档")
            return True

        except Exception as e:
            self.logger.error(f"更新索引失败: {str(e)}")
            return False

    async def delete_from_index(self, document_ids: List[str]) -> bool:
        """从索引中删除文档"""
        try:
            # 找到要删除的文档索引
            indices_to_keep = []
            docs_to_keep = []

            for i, doc in enumerate(self.indexed_documents):
                if doc.id not in document_ids:
                    indices_to_keep.append(i)
                    docs_to_keep.append(doc)

            # 更新文档列表
            self.indexed_documents = docs_to_keep

            # 更新嵌入矩阵
            if self.document_embeddings is not None and indices_to_keep:
                self.document_embeddings = self.document_embeddings[indices_to_keep]

            self.logger.info(f"成功删除 {len(document_ids)} 个文档")
            return True

        except Exception as e:
            self.logger.error(f"删除文档失败: {str(e)}")
            return False

    async def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """根据ID获取文档"""
        for doc in self.indexed_documents:
            if doc.id == document_id:
                return doc
        return None

    async def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        if self.embedding_model:
            return self.embedding_model.get_dimension()
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            'indexed_documents': len(self.indexed_documents),
            'embedding_dimension': self.get_embedding_dimension(),
            'cache_enabled': self.cache_manager is not None,
            'index_size_mb': self.document_embeddings.nbytes / (1024 * 1024) if self.document_embeddings is not None else 0
        }


class FaissVectorRetriever(VectorRetriever):
    """基于FAISS的向量检索器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index = None
        self.faiss_index_type = self.get_config_value('faiss_index_type', 'flat')

    def _create_embedding_model(self, config: Dict[str, Any]) -> IEmbeddingModel:
        """创建FAISS兼容的嵌入模型"""
        from .embedding import SentenceTransformersEmbedding
        return SentenceTransformersEmbedding(config)

    def _create_cache_manager(self, config: Dict[str, Any]) -> ICacheManager:
        """创建缓存管理器"""
        from ..cache import MemoryCacheManager
        return MemoryCacheManager(config)

    async def _do_initialize(self) -> None:
        """初始化FAISS索引"""
        await super()._do_initialize()
        self._initialize_faiss_index()

    def _initialize_faiss_index(self):
        """初始化FAISS索引"""
        try:
            import faiss

            dimension = self.get_embedding_dimension()
            if dimension == 0:
                raise ValueError("嵌入维度为0，无法初始化FAISS索引")

            # 根据配置选择索引类型
            if self.faiss_index_type == 'flat':
                self.index = faiss.IndexFlatIP(dimension)  # 内积索引
            elif self.faiss_index_type == 'ivf':
                nlist = self.get_config_value('nlist', 100)
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            elif self.faiss_index_type == 'hnsw':
                M = self.get_config_value('M', 16)
                self.index = faiss.IndexHNSWFlat(dimension, M)
            else:
                self.index = faiss.IndexFlatIP(dimension)

            self.logger.info(f"FAISS索引初始化完成，类型: {self.faiss_index_type}, 维度: {dimension}")

        except ImportError:
            self.logger.warning("FAISS未安装，使用numpy相似度计算")
            self.index = None
        except Exception as e:
            self.logger.error(f"FAISS索引初始化失败: {str(e)}")
            self.index = None

    def _compute_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """计算相似度"""
        if self.index is not None:
            # 使用FAISS计算相似度
            try:
                # 归一化向量（用于余弦相似度）
                faiss.normalize_L2(query_vector)
                faiss.normalize_L2(self.document_embeddings)

                # 搜索
                scores, _ = self.index.search(query_vector, len(self.indexed_documents))
                return scores
            except Exception as e:
                self.logger.error(f"FAISS搜索失败: {str(e)}")
                # 降级到numpy计算
                pass

        # 使用numpy计算余弦相似度
        query_norm = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        doc_norms = self.document_embeddings / np.linalg.norm(self.document_embeddings, axis=1, keepdims=True)
        similarities = np.dot(query_norm, doc_norms.T)

        return similarities

    async def update_index(self, documents: List[Document]) -> bool:
        """更新FAISS索引"""
        success = await super().update_index(documents)

        if success and self.index is not None:
            # 重建FAISS索引
            try:
                import faiss

                # 重新训练IVF索引（如果使用）
                if hasattr(self.index, 'train') and self.document_embeddings.shape[0] > 0:
                    self.index.train(self.document_embeddings)

                # 添加向量到索引
                self.index.reset()
                if self.document_embeddings.shape[0] > 0:
                    # 归一化向量
                    normalized_embeddings = self.document_embeddings.copy()
                    faiss.normalize_L2(normalized_embeddings)
                    self.index.add(normalized_embeddings)

                self.logger.info("FAISS索引更新完成")

            except Exception as e:
                self.logger.error(f"FAISS索引更新失败: {str(e)}")
                return False

        return success