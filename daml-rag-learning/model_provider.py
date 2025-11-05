#!/usr/bin/env python3
"""
DAML-RAG Framework 模型提供者模块
实现教师模型和学生模型的双模型机制
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import openai
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型枚举"""
    TEACHER = "teacher"  # 教师模型，高质量推理
    STUDENT = "student"  # 学生模型，成本优化


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    model_type: ModelType
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    cost_per_token: float = 0.0
    context_window: int = 4096


@dataclass
class GenerationRequest:
    """生成请求"""
    prompt: str
    context: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None


@dataclass
class GenerationResponse:
    """生成响应"""
    content: str
    model_used: str
    tokens_used: int
    cost: float
    execution_time: float
    cache_hit: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ModelProvider(ABC):
    """模型提供者抽象基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.model_type = config.model_type
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """初始化模型提供者"""
        pass

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest):
        """流式生成文本"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

    async def close(self):
        """关闭连接"""
        if self._session:
            await self._session.close()
            self._session = None

    def _calculate_cost(self, tokens: int) -> float:
        """计算成本"""
        return tokens * self.config.cost_per_token

    def _merge_context(self, prompt: str, context: Optional[List[str]]) -> str:
        """合并上下文"""
        if not context:
            return prompt

        context_str = "\n".join([f"- {c}" for c in context])
        return f"""背景信息：
{context_str}

请基于以上背景信息回答以下问题：
{prompt}"""

    async def _ensure_session(self):
        """确保会话存在"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)


class OpenAIProvider(ModelProvider):
    """OpenAI模型提供者"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[openai.AsyncOpenAI] = None

    async def initialize(self) -> None:
        """初始化OpenAI客户端"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )

        # 测试连接
        await self.health_check()
        self._initialized = True
        logger.info(f"OpenAI provider initialized with model: {self.model_name}")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # 合并上下文
        full_prompt = self._merge_context(request.prompt, request.context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": request.system_prompt or "你是一个专业的健身教练"},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature,
                stop=request.stop_sequences
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = self._calculate_cost(tokens_used)
            execution_time = time.time() - start_time

            return GenerationResponse(
                content=content,
                model_used=self.model_name,
                tokens_used=tokens_used,
                cost=cost,
                execution_time=execution_time,
                cache_hit=False,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model_response": response.model
                }
            )

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def generate_stream(self, request: GenerationRequest):
        """流式生成文本"""
        if not self._initialized:
            await self.initialize()

        full_prompt = self._merge_context(request.prompt, request.context)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": request.system_prompt or "你是一个专业的健身教练"},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=request.temperature or self.config.temperature,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI stream generation error: {e}")
            raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


class DeepSeekProvider(ModelProvider):
    """DeepSeek模型提供者（教师模型）"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = config.api_key

    async def initialize(self) -> None:
        """初始化DeepSeek提供者"""
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")

        await self._ensure_session()
        await self.health_check()
        self._initialized = True
        logger.info(f"DeepSeek provider initialized with model: {self.model_name}")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # 合并上下文
        full_prompt = self._merge_context(request.prompt, request.context)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt or "你是一个专业的健身教练，请提供详细、准确、安全的健身指导"},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "stop": request.stop_sequences
        }

        try:
            async with self._session.post(
                f"{self.config.api_base or 'https://api.deepseek.com'}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()

                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                cost = self._calculate_cost(tokens_used)
                execution_time = time.time() - start_time

                return GenerationResponse(
                    content=content,
                    model_used=self.model_name,
                    tokens_used=tokens_used,
                    cost=cost,
                    execution_time=execution_time,
                    cache_hit=False,
                    metadata={
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "model_response": data.get("model")
                    }
                )

        except Exception as e:
            logger.error(f"DeepSeek generation error: {e}")
            raise

    async def generate_stream(self, request: GenerationRequest):
        """流式生成文本"""
        if not self._initialized:
            await self.initialize()

        full_prompt = self._merge_context(request.prompt, request.context)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt or "你是一个专业的健身教练"},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": request.temperature or self.config.temperature,
            "stream": True
        }

        try:
            async with self._session.post(
                f"{self.config.api_base or 'https://api.deepseek.com'}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if data["choices"][0]["delta"].get("content"):
                                yield data["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"DeepSeek stream generation error: {e}")
            raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }

            async with self._session.post(
                f"{self.config.api_base or 'https://api.deepseek.com'}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"DeepSeek health check failed: {e}")
            return False


class OllamaProvider(ModelProvider):
    """Ollama本地模型提供者（学生模型）"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.api_base or "http://localhost:11434"

    async def initialize(self) -> None:
        """初始化Ollama提供者"""
        await self._ensure_session()

        # 检查模型是否可用
        if not await self._check_model_available():
            raise ValueError(f"Model {self.model_name} not found in Ollama")

        await self.health_check()
        self._initialized = True
        logger.info(f"Ollama provider initialized with model: {self.model_name}")

    async def _check_model_available(self) -> bool:
        """检查模型是否在Ollama中可用"""
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                data = await response.json()

                models = [model["name"] for model in data.get("models", [])]
                return any(self.model_name in model for model in models)

        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
            return False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # 合并上下文
        full_prompt = self._merge_context(request.prompt, request.context)

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "system": request.system_prompt or "你是一个专业的健身教练",
            "options": {
                "temperature": request.temperature or self.config.temperature,
                "num_predict": request.max_tokens or self.config.max_tokens,
                "stop": request.stop_sequences or []
            }
        }

        try:
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()

                content = data.get("response", "")
                # Ollama doesn't provide token count, estimate it
                tokens_used = len(content) // 4  # Rough estimate
                cost = self._calculate_cost(tokens_used)
                execution_time = time.time() - start_time

                return GenerationResponse(
                    content=content,
                    model_used=self.model_name,
                    tokens_used=tokens_used,
                    cost=cost,
                    execution_time=execution_time,
                    cache_hit=False,
                    metadata={
                        "done": data.get("done", False),
                        "total_duration": data.get("total_duration", 0),
                        "load_duration": data.get("load_duration", 0)
                    }
                )

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def generate_stream(self, request: GenerationRequest):
        """流式生成文本"""
        if not self._initialized:
            await self.initialize()

        full_prompt = self._merge_context(request.prompt, request.context)

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "system": request.system_prompt or "你是一个专业的健身教练",
            "options": {
                "temperature": request.temperature or self.config.temperature,
                "num_predict": request.max_tokens or self.config.max_tokens,
                "stop": request.stop_sequences or []
            }
        }

        try:
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("response"):
                                yield data["response"]
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Ollama stream generation error: {e}")
            raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


class CachedModelProvider(ModelProvider):
    """带缓存的模型提供者装饰器"""

    def __init__(self, provider: ModelProvider, cache_manager):
        self.provider = provider
        self.cache_manager = cache_manager
        self.config = provider.config
        self.model_name = provider.model_name
        self.model_type = provider.model_type

    async def initialize(self) -> None:
        """初始化被装饰的提供者"""
        await self.provider.initialize()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """带缓存的生成"""
        # 生成缓存键
        cache_key = self._generate_cache_key(request)

        # 尝试从缓存获取
        cached_response = await self.cache_manager.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for model: {self.model_name}")
            cached_response.cache_hit = True
            return cached_response

        # 缓存未命中，调用原始提供者
        response = await self.provider.generate(request)

        # 存储到缓存
        await self.cache_manager.set(
            cache_key,
            response,
            ttl=3600  # 1小时缓存
        )

        return response

    async def generate_stream(self, request: GenerationRequest):
        """流式生成（不缓存）"""
        async for chunk in self.provider.generate_stream(request):
            yield chunk

    async def health_check(self) -> bool:
        """健康检查"""
        return await self.provider.health_check()

    async def close(self):
        """关闭连接"""
        await self.provider.close()

    def _generate_cache_key(self, request: GenerationRequest) -> str:
        """生成缓存键"""
        import hashlib

        key_data = {
            "prompt": request.prompt,
            "context": request.context,
            "system_prompt": request.system_prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "model": self.model_name
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


class ModelManager:
    """模型管理器 - 负责教师模型和学生模型的智能选择"""

    def __init__(
        self,
        teacher_provider: ModelProvider,
        student_provider: ModelProvider,
        complexity_threshold: float = 0.7
    ):
        self.teacher_provider = teacher_provider
        self.student_provider = student_provider
        self.complexity_threshold = complexity_threshold

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "teacher_requests": 0,
            "student_requests": 0,
            "cache_hits": 0,
            "total_cost": 0.0,
            "total_tokens": 0
        }

    async def initialize(self) -> None:
        """初始化所有模型提供者"""
        await asyncio.gather(
            self.teacher_provider.initialize(),
            self.student_provider.initialize()
        )
        logger.info("Model manager initialized with teacher and student models")

    async def generate(
        self,
        request: GenerationRequest,
        force_teacher: bool = False
    ) -> GenerationResponse:
        """智能选择模型生成文本"""
        self.stats["total_requests"] += 1

        # 强制使用教师模型
        if force_teacher:
            self.stats["teacher_requests"] += 1
            response = await self.teacher_provider.generate(request)
            self._update_stats(response)
            return response

        # 分析查询复杂度决定使用哪个模型
        complexity = self._analyze_complexity(request)

        if complexity >= self.complexity_threshold:
            # 高复杂度，使用教师模型
            logger.debug(f"Using teacher model for complexity {complexity:.2f}")
            self.stats["teacher_requests"] += 1
            response = await self.teacher_provider.generate(request)
        else:
            # 低复杂度，使用学生模型
            logger.debug(f"Using student model for complexity {complexity:.2f}")
            self.stats["student_requests"] += 1
            response = await self.student_provider.generate(request)

        self._update_stats(response)
        return response

    async def generate_stream(
        self,
        request: GenerationRequest,
        force_teacher: bool = False
    ):
        """智能选择模型流式生成"""
        complexity = self._analyze_complexity(request)

        if force_teacher or complexity >= self.complexity_threshold:
            self.stats["teacher_requests"] += 1
            async for chunk in self.teacher_provider.generate_stream(request):
                yield chunk
        else:
            self.stats["student_requests"] += 1
            async for chunk in self.student_provider.generate_stream(request):
                yield chunk

    def _analyze_complexity(self, request: GenerationRequest) -> float:
        """分析查询复杂度"""
        complexity = 0.0

        # 基于查询长度
        prompt_length = len(request.prompt)
        if prompt_length > 200:
            complexity += 0.2
        elif prompt_length > 100:
            complexity += 0.1

        # 基于上下文数量
        if request.context:
            if len(request.context) > 5:
                complexity += 0.3
            elif len(request.context) > 2:
                complexity += 0.15

        # 基于关键词
        complex_keywords = [
            "制定", "设计", "分析", "评估", "诊断",
            "康复", "损伤", "疾病", "营养计划", "周期化",
            "prog", "设计", "analyze", "rehab", "injury"
        ]

        keyword_count = sum(1 for keyword in complex_keywords if keyword in request.prompt)
        complexity += min(keyword_count * 0.15, 0.4)

        # 基于问题类型
        question_patterns = ["如何", "怎样", "什么原因", "为什么", "how to", "why"]
        if any(pattern in request.prompt for pattern in question_patterns):
            complexity += 0.1

        return min(complexity, 1.0)

    def _update_stats(self, response: GenerationResponse):
        """更新统计信息"""
        if response.cache_hit:
            self.stats["cache_hits"] += 1

        self.stats["total_cost"] += response.cost
        self.stats["total_tokens"] += response.tokens_used

    async def health_check(self) -> Dict[str, bool]:
        """检查所有模型健康状态"""
        return {
            "teacher": await self.teacher_provider.health_check(),
            "student": await self.student_provider.health_check()
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats["total_requests"]
        if total == 0:
            return self.stats

        stats_with_ratios = self.stats.copy()
        stats_with_ratios.update({
            "teacher_ratio": self.stats["teacher_requests"] / total,
            "student_ratio": self.stats["student_requests"] / total,
            "cache_hit_ratio": self.stats["cache_hits"] / total,
            "avg_cost_per_request": self.stats["total_cost"] / total,
            "avg_tokens_per_request": self.stats["total_tokens"] / total
        })

        return stats_with_ratios

    async def close(self):
        """关闭所有连接"""
        await asyncio.gather(
            self.teacher_provider.close(),
            self.student_provider.close()
        )