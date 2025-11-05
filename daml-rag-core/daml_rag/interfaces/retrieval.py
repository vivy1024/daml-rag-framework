"""
检索相关接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio

from ..models import (
    RetrievalResult,
    Document,
    Entity,
    Relation,
    Path,
    SimilarityResult,
    ValidationResult,
)


class IRetriever(ABC):
    """检索器抽象接口"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行检索"""
        pass

    @abstractmethod
    async def batch_retrieve(self, queries: List[str],
                           top_k: int = 5) -> List[RetrievalResult]:
        """批量检索"""
        pass

    @abstractmethod
    async def update_index(self, documents: List[Document]) -> bool:
        """更新索引"""
        pass

    @abstractmethod
    async def delete_from_index(self, document_ids: List[str]) -> bool:
        """从索引中删除文档"""
        pass

    @abstractmethod
    async def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """根据ID获取文档"""
        pass


class IVectorRetriever(IRetriever):
    """向量检索器接口"""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """文本向量化"""
        pass

    @abstractmethod
    async def similarity_search(self, query_vector: List[float],
                              top_k: int = 5,
                              filters: Optional[Dict[str, Any]] = None) -> List[SimilarityResult]:
        """相似度搜索"""
        pass

    @abstractmethod
    async def hybrid_search(self, query: str, query_vector: List[float],
                          top_k: int = 5, alpha: float = 0.7) -> List[SimilarityResult]:
        """混合搜索（文本+向量）"""
        pass

    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        pass


class IKnowledgeGraphRetriever(IRetriever):
    """知识图谱检索器接口"""

    @abstractmethod
    async def query_entities(self, query: str,
                           entity_types: Optional[List[str]] = None) -> List[Entity]:
        """实体查询"""
        pass

    @abstractmethod
    async def find_relations(self, source: str, target: str,
                           relation_types: Optional[List[str]] = None) -> List[Relation]:
        """关系查询"""
        pass

    @abstractmethod
    async def path_query(self, start: str, end: str,
                        max_depth: int = 3,
                        relation_types: Optional[List[str]] = None) -> List[Path]:
        """路径查询"""
        pass

    @abstractmethod
    async def neighbor_query(self, entity_id: str,
                           depth: int = 1,
                           direction: str = "both") -> List[Entity]:
        """邻居查询"""
        pass

    @abstractmethod
    async def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        pass

    @abstractmethod
    async def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        pass

    @abstractmethod
    async def cypher_query(self, query: str,
                         parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Cypher查询"""
        pass


class IRuleFilter(ABC):
    """规则过滤器接口"""

    @abstractmethod
    async def filter(self, results: RetrievalResult,
                    context: Dict[str, Any]) -> RetrievalResult:
        """结果过滤"""
        pass

    @abstractmethod
    async def score(self, result: RetrievalResult,
                   context: Dict[str, Any]) -> float:
        """结果评分"""
        pass

    @abstractmethod
    async def validate_params(self, params: Dict[str, Any]) -> ValidationResult:
        """参数验证"""
        pass

    @abstractmethod
    def get_filter_rules(self) -> List[str]:
        """获取过滤规则列表"""
        pass


class ICacheManager(ABC):
    """缓存管理器接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any,
                 ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass


class IRetrievalPipeline(ABC):
    """检索流水线接口"""

    @abstractmethod
    async def process(self, query: str,
                     context: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """处理查询"""
        pass

    @abstractmethod
    def add_retriever(self, retriever: IRetriever,
                     weight: float = 1.0) -> None:
        """添加检索器"""
        pass

    @abstractmethod
    def add_filter(self, filter_rule: IRuleFilter) -> None:
        """添加过滤器"""
        pass

    @abstractmethod
    async def warm_up(self) -> bool:
        """预热"""
        pass


class IDocumentProcessor(ABC):
    """文档处理器接口"""

    @abstractmethod
    async def process(self, document: Document) -> Document:
        """处理文档"""
        pass

    @abstractmethod
    async def batch_process(self, documents: List[Document]) -> List[Document]:
        """批量处理文档"""
        pass

    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        pass


class IEmbeddingModel(ABC):
    """嵌入模型接口"""

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """编码文本"""
        pass

    @abstractmethod
    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """检查模型是否可用"""
        pass