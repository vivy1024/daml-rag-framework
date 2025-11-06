#!/usr/bin/env python3
"""
DAML-RAG健身 框架 FAISS向量检索实现
Facebook AI相似度搜索本地部署
"""

import asyncio
import logging
import pickle
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import numpy as np
from datetime import datetime
import faiss

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..base import BaseRetriever, RetrievalResult, Document

logger = logging.getLogger(__name__)


@dataclass
class FAISSConfig:
    """FAISS配置"""
    index_type: str = "flat"  # flat, ivf, hnsw
    vector_size: int = 768
    metric_type: str = "cosine"  # cosine, l2, inner_product
    nlist: int = 100  # for IVF indexes
    nprobe: int = 10  # for IVF search
    M: int = 16  # for HNSW
    efConstruction: int = 200  # for HNSW
    efSearch: int = 50  # for HNSW
    index_path: Optional[str] = None
    save_index: bool = True
    use_gpu: bool = False
    gpu_id: int = 0

    def __post_init__(self):
        # 验证配置
        valid_metrics = ["cosine", "l2", "inner_product"]
        if self.metric_type not in valid_metrics:
            raise ValueError(f"metric_type must be one of {valid_metrics}")

        valid_index_types = ["flat", "ivf", "hnsw"]
        if self.index_type not in valid_index_types:
            raise ValueError(f"index_type must be one of {valid_index_types}")


class FAISSVectorRetriever(BaseRetriever):
    """FAISS向量检索器"""

    def __init__(self, config: FAISSConfig):
        super().__init__(config)
        self.config = config
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.doc_id_to_index: Dict[str, int] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """初始化FAISS索引"""
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not available. Install with: pip install faiss-cpu or faiss-gpu"
            )

        try:
            # 创建FAISS索引
            self.index = await self._create_index()

            # 尝试加载已有索引
            if self.config.index_path and os.path.exists(self.config.index_path):
                await self._load_index()
                logger.info(f"Loaded existing index from {self.config.index_path}")
            else:
                logger.info("Created new FAISS index")

            self._initialized = True
            logger.info(f"FAISS vector retriever initialized: {self.config.index_type}")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise

    async def _create_index(self) -> faiss.Index:
        """创建FAISS索引"""
        dimension = self.config.vector_size

        if self.config.use_gpu:
            try:
                # GPU索引
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, self.config.gpu_id)
                return self._create_cpu_index(gpu_index)
            except Exception as e:
                logger.warning(f"GPU not available, falling back to CPU: {e}")
                self.config.use_gpu = False

        return self._create_cpu_index()

    def _create_cpu_index(self, index: Optional[faiss.Index] = None) -> faiss.Index:
        """创建CPU索引"""
        dimension = self.config.vector_size

        if self.config.index_type == "flat":
            if self.config.metric_type == "l2":
                index = faiss.IndexFlatL2(dimension)
            elif self.config.metric_type == "cosine":
                # 余弦相似度需要归一化
                index = faiss.IndexFlatIP(dimension)
            elif self.config.metric_type == "inner_product":
                index = faiss.IndexFlatIP(dimension)

        elif self.config.index_type == "ivf":
            if self.config.metric_type == "l2":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.metric_type == "cosine":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)
            elif self.config.metric_type == "inner_product":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist)

        elif self.config.index_type == "hnsw":
            if self.config.metric_type == "l2":
                index = faiss.IndexHNSWFlat(dimension, self.config.M)
            elif self.config.metric_type == "cosine":
                index = faiss.IndexHNSWFlat(dimension, self.config.M)
                # HNSW不直接支持余弦，需要使用内积和归一化
                index.hnsw.efConstruction = self.config.efConstruction
            elif self.config.metric_type == "inner_product":
                index = faiss.IndexHNSWFlat(dimension, self.config.M)
                index.hnsw.efConstruction = self.config.efConstruction

        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        return index

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到FAISS索引"""
        if not self._initialized:
            await self.initialize()

        if not documents:
            return []

        try:
            vectors = []
            doc_ids = []

            for doc in documents:
                if doc.vector is None:
                    logger.warning(f"Document {doc.id} has no vector, skipping")
                    continue

                # 归一化向量（用于余弦相似度）
                vector = np.array(doc.vector, dtype=np.float32)
                if self.config.metric_type == "cosine":
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm

                vectors.append(vector)
                doc_ids.append(doc.id)

            if not vectors:
                return []

            # 转换为numpy数组
            vectors_array = np.vstack(vectors).astype(np.float32)

            # 添加到索引
            start_idx = len(self.documents)

            if self.config.index_type == "ivf" and not self.index.is_trained:
                # IVF索引需要训练
                logger.info("Training IVF index...")
                self.index.train(vectors_array)
                logger.info("IVF index training completed")

            self.index.add(vectors_array)

            # 更新文档列表和映射
            for i, doc in enumerate(documents):
                if doc.vector is not None:
                    self.documents.append(doc)
                    self.doc_id_to_index[doc.id] = start_idx + i

            # 保存索引
            if self.config.save_index and self.config.index_path:
                await self._save_index()

            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise

    async def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_condition: Optional[Dict[str, Any]] = None,
        include_payload: bool = True,
        include_vectors: bool = False
    ) -> RetrievalResult:
        """FAISS向量检索"""
        if not self._initialized:
            await self.initialize()

        try:
            # 准备查询向量
            query_vector = np.array(query_vector, dtype=np.float32)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)

            # 归一化查询向量（用于余弦相似度）
            if self.config.metric_type == "cosine":
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm

            # 设置搜索参数
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.config.nprobe
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = self.config.efSearch

            # 执行搜索
            search_k = min(top_k * 2, len(self.documents))  # 搜索更多用于过滤
            scores, indices = self.index.search(query_vector, search_k)

            # 处理结果
            documents = []
            final_scores = []
            metadatas = []

            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue

                if idx >= len(self.documents):
                    continue

                doc = self.documents[idx]

                # 应用分数阈值
                if score < score_threshold:
                    continue

                # 应用过滤条件
                if filter_condition and not self._apply_filter(doc, filter_condition):
                    continue

                documents.append(doc)
                final_scores.append(score)

                # 构建metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata.update({
                    "faiss_score": float(score),
                    "faiss_index": int(idx),
                    "retrieval_rank": len(documents)
                })
                metadatas.append(metadata)

                # 限制结果数量
                if len(documents) >= top_k:
                    break

            return RetrievalResult(
                query="",
                documents=documents,
                scores=final_scores,
                metadatas=metadatas,
                retrieval_method="faiss_vector_search",
                total_found=len(documents),
                search_time=0.0  # FAISS不提供精确时间
            )

        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            raise

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """删除文档（FAISS不支持删除，需要重建索引）"""
        logger.warning("FAISS does not support individual document deletion. Need to rebuild index.")
        return False

    async def update_document(self, doc_id: str, document: Document) -> bool:
        """更新文档（FAISS不支持更新，需要重建索引）"""
        logger.warning("FAISS does not support individual document updates. Need to rebuild index.")
        return False

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """获取单个文档"""
        if doc_id in self.doc_id_to_index:
            idx = self.doc_id_to_index[doc_id]
            if idx < len(self.documents):
                return self.documents[idx]
        return None

    async def rebuild_index(self, documents: List[Document] = None) -> bool:
        """重建索引"""
        try:
            # 保存当前文档
            old_documents = self.documents.copy()

            # 使用新文档或旧文档重建
            docs_to_use = documents if documents is not None else old_documents

            # 重置索引
            self.index = await self._create_index()
            self.documents.clear()
            self.doc_id_to_index.clear()

            # 重新添加文档
            if docs_to_use:
                await self.add_documents(docs_to_use)

            logger.info("FAISS index rebuilt successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
            return False

    async def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        if not self._initialized:
            await self.initialize()

        info = {
            "index_type": self.config.index_type,
            "vector_size": self.config.vector_size,
            "metric_type": self.config.metric_type,
            "total_vectors": len(self.documents),
            "is_trained": getattr(self.index, 'is_trained', True),
            "ntotal": getattr(self.index, 'ntotal', 0)
        }

        # 添加特定索引类型的信息
        if hasattr(self.index, 'nlist'):
            info["nlist"] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            info["nprobe"] = self.index.nprobe
        if hasattr(self.index, 'M'):
            info["M"] = self.index.M
        if hasattr(self.index, 'hnsw'):
            info["efConstruction"] = self.index.hnsw.efConstruction
            info["efSearch"] = self.index.hnsw.efSearch

        return info

    def _apply_filter(self, document: Document, filter_condition: Dict[str, Any]) -> bool:
        """应用过滤条件"""
        if not document.metadata:
            return filter_condition == {}

        for field, condition in filter_condition.items():
            if field not in document.metadata:
                return False

            value = document.metadata[field]

            if isinstance(condition, dict):
                if "eq" in condition and value != condition["eq"]:
                    return False
                if "ne" in condition and value == condition["ne"]:
                    return False
                if "in" in condition and value not in condition["in"]:
                    return False
                if "nin" in condition and value in condition["nin"]:
                    return False
                if "gt" in condition and not (value > condition["gt"]):
                    return False
                if "gte" in condition and not (value >= condition["gte"]):
                    return False
                if "lt" in condition and not (value < condition["lt"]):
                    return False
                if "lte" in condition and not (value <= condition["lte"]):
                    return False
            elif value != condition:
                return False

        return True

    async def _save_index(self) -> None:
        """保存索引到文件"""
        if not self.config.index_path:
            return

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config.index_path), exist_ok=True)

            # 保存FAISS索引
            faiss.write_index(self.index, self.config.index_path)

            # 保存元数据
            metadata_path = self.config.index_path.replace(".index", ".metadata")
            metadata = {
                "config": asdict(self.config),
                "documents_count": len(self.documents),
                "doc_id_to_index": self.doc_id_to_index,
                "documents": [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in self.documents
                ]
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved FAISS index to {self.config.index_path}")

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    async def _load_index(self) -> None:
        """从文件加载索引"""
        if not self.config.index_path or not os.path.exists(self.config.index_path):
            return

        try:
            # 加载FAISS索引
            self.index = faiss.read_index(self.config.index_path)

            # 加载元数据
            metadata_path = self.config.index_path.replace(".index", ".metadata")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                self.documents = [
                    Document(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        metadata=doc_data["metadata"]
                    )
                    for doc_data in metadata["documents"]
                ]
                self.doc_id_to_index = metadata["doc_id_to_index"]

            logger.info(f"Loaded FAISS index from {self.config.index_path}")

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")

    async def health_check(self) -> bool:
        """健康检查"""
        return self._initialized and self.index is not None

    async def close(self):
        """关闭连接"""
        if self.config.save_index and self.config.index_path:
            await self._save_index()
        logger.info("FAISS retriever closed")


class FAISSVectorManager:
    """FAISS向量管理器 - 提供高级管理功能"""

    def __init__(self, config: FAISSConfig):
        self.config = config
        self.retriever: Optional[FAISSVectorRetriever] = None

    async def initialize(self) -> None:
        """初始化管理器"""
        self.retriever = FAISSVectorRetriever(self.config)
        await self.retriever.initialize()

    async def create_training_data(
        self,
        documents: List[Document],
        training_ratio: float = 0.1
    ) -> Tuple[List[Document], List[Document]]:
        """创建训练数据"""
        if not self.retriever:
            await self.initialize()

        total_docs = len(documents)
        training_size = int(total_docs * training_ratio)

        indices = np.random.permutation(total_docs)
        training_indices = indices[:training_size]
        test_indices = indices[training_size:]

        training_docs = [documents[i] for i in training_indices]
        test_docs = [documents[i] for i in test_indices]

        logger.info(f"Created {len(training_docs)} training and {len(test_docs)} test documents")
        return training_docs, test_docs

    async def evaluate_index(
        self,
        test_queries: List[np.ndarray],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """评估索引性能"""
        if not self.retriever:
            await self.initialize()

        results = {f"recall@{k}": 0.0 for k in k_values}
        total_queries = len(test_queries)

        for k in k_values:
            recall_sum = 0.0
            for query_vec, true_doc_ids in zip(test_queries, ground_truth):
                search_result = await self.retriever.search(query_vec, top_k=k)
                retrieved_ids = [doc.id for doc in search_result.documents]

                # 计算recall@k
                intersection = len(set(retrieved_ids) & set(true_doc_ids))
                recall = intersection / len(true_doc_ids) if true_doc_ids else 0.0
                recall_sum += recall

            results[f"recall@{k}"] = recall_sum / total_queries

        return results

    async def optimize_index_parameters(
        self,
        documents: List[Document],
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """优化索引参数"""
        best_config = asdict(self.config)
        best_score = 0.0

        # 生成参数组合
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combination in itertools.product(*param_values):
            # 创建临时配置
            temp_config = FAISSConfig(**{**asdict(self.config), **dict(zip(param_names, combination))})

            # 创建临时检索器
            temp_retriever = FAISSVectorRetriever(temp_config)
            await temp_retriever.initialize()

            # 添加文档
            await temp_retriever.add_documents(documents)

            # 简单评估（这里可以根据需要扩展）
            info = await temp_retriever.get_index_info()
            score = info.get("total_vectors", 0)  # 简单示例：以向量数量为指标

            if score > best_score:
                best_score = score
                best_config = asdict(temp_config)

            logger.info(f"Tested parameters: {dict(zip(param_names, combination))}, Score: {score}")

        logger.info(f"Best configuration: {best_config}, Score: {best_score}")
        return {"best_config": best_config, "best_score": best_score}

    async def close(self):
        """关闭管理器"""
        if self.retriever:
            await self.retriever.close()
            self.retriever = None