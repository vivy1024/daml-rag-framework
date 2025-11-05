"""
嵌入模型实现
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer

from daml_rag.interfaces import IEmbeddingModel
from daml_rag.base import ConfigurableComponent


class SentenceTransformersEmbedding(IEmbeddingModel, ConfigurableComponent):
    """SentenceTransformers嵌入模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = self.get_config_value('model_name', 'BAAI/bge-base-zh-v1.5')
        self.device = self.get_config_value('device', 'cpu')
        self.batch_size = self.get_config_value('batch_size', 32)
        self.model: Optional[SentenceTransformer] = None

    async def _do_initialize(self) -> None:
        """初始化模型"""
        try:
            self.logger.info(f"加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # 预热模型
            test_text = ["测试文本"]
            await self.encode(test_text)

            self.logger.info(f"嵌入模型加载完成，维度: {self.get_dimension()}")

        except Exception as e:
            self.logger.error(f"嵌入模型加载失败: {str(e)}")
            raise

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """编码文本"""
        if not self.model:
            raise RuntimeError("模型未初始化")

        try:
            # 批量编码
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings.extend(batch_embeddings.tolist())

            return embeddings

        except Exception as e:
            self.logger.error(f"文本编码失败: {str(e)}")
            raise

    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        embeddings = await self.encode([text])
        return embeddings[0]

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'dimension': self.get_dimension(),
            'batch_size': self.batch_size,
            'max_seq_length': getattr(self.model, 'max_seq_length', None) if self.model else None
        }

    async def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            if not self.model:
                return False
            # 测试编码
            await self.encode_single("测试")
            return True
        except Exception:
            return False


class OpenAIEmbedding(IEmbeddingModel, ConfigurableComponent):
    """OpenAI嵌入模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self.get_config_value('api_key')
        self.model_name = self.get_config_value('model_name', 'text-embedding-ada-002')
        self.base_url = self.get_config_value('base_url', 'https://api.openai.com/v1')
        self.max_retries = self.get_config_value('max_retries', 3)
        self.timeout = self.get_config_value('timeout', 30)

    async def _do_initialize(self) -> None:
        """初始化OpenAI客户端"""
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self.logger.info(f"OpenAI嵌入客户端初始化完成，模型: {self.model_name}")
        except ImportError:
            raise ImportError("需要安装openai包: pip install openai")
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败: {str(e)}")
            raise

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """编码文本"""
        if not hasattr(self, 'client'):
            raise RuntimeError("客户端未初始化")

        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            self.logger.error(f"OpenAI文本编码失败: {str(e)}")
            raise

    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        embeddings = await self.encode([text])
        return embeddings[0]

    def get_dimension(self) -> int:
        """获取向量维度"""
        # OpenAI嵌入模型的维度
        dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
        }
        return dimensions.get(self.model_name, 1536)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'base_url': self.base_url,
            'dimension': self.get_dimension(),
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }

    async def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            await self.encode_single("test")
            return True
        except Exception:
            return False


class CacheEmbedding(IEmbeddingModel):
    """带缓存的嵌入模型装饰器"""

    def __init__(self, base_model: IEmbeddingModel, cache_manager):
        self.base_model = base_model
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """编码文本（带缓存）"""
        embeddings = []
        texts_to_encode = []
        cache_keys = []

        # 检查缓存
        for text in texts:
            cache_key = f"embedding:{hash(text)}"
            cache_keys.append(cache_key)

            cached_embedding = await self.cache_manager.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                texts_to_encode.append(text)

        # 编码未缓存的文本
        if texts_to_encode:
            try:
                new_embeddings = await self.base_model.encode(texts_to_encode)

                # 存储到缓存
                cache_index = 0
                for i, embedding in enumerate(embeddings):
                    if embedding is None:
                        cache_key = cache_keys[i]
                        new_embedding = new_embeddings[cache_index]
                        await self.cache_manager.set(cache_key, new_embedding, ttl=3600)
                        embeddings[i] = new_embedding
                        cache_index += 1

            except Exception as e:
                self.logger.error(f"嵌入编码失败: {str(e)}")
                # 对于失败的编码，使用零向量
                zero_embedding = [0.0] * self.get_dimension()
                for i, embedding in enumerate(embeddings):
                    if embedding is None:
                        embeddings[i] = zero_embedding

        return embeddings

    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        embeddings = await self.encode([text])
        return embeddings[0]

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.base_model.get_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.base_model.get_model_info()
        info['cached'] = True
        return info

    async def is_available(self) -> bool:
        """检查模型是否可用"""
        return await self.base_model.is_available()


class EmbeddingModelFactory:
    """嵌入模型工厂"""

    @staticmethod
    def create_model(config: Dict[str, Any]) -> IEmbeddingModel:
        """创建嵌入模型"""
        model_type = config.get('type', 'sentence_transformers')

        if model_type == 'sentence_transformers':
            return SentenceTransformersEmbedding(config)
        elif model_type == 'openai':
            return OpenAIEmbedding(config)
        else:
            raise ValueError(f"不支持的嵌入模型类型: {model_type}")

    @staticmethod
    def create_cached_model(base_model: IEmbeddingModel, cache_config: Dict[str, Any]) -> IEmbeddingModel:
        """创建带缓存的嵌入模型"""
        from .cache import MemoryCacheManager
        cache_manager = MemoryCacheManager(cache_config)
        return CacheEmbedding(base_model, cache_manager)