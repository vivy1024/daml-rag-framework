#!/usr/bin/env python3
"""
DAML-RAG 框架 学习模块
"""

from .memory import (
    MemoryManager,
    InMemoryManager,
    RedisMemoryManager,
    Experience,
    Feedback,
    FeedbackType,
    RetrievalResult
)

from .model_provider import (
    ModelProvider,
    DeepSeekProvider,
    OllamaProvider,
    OpenAIProvider,
    CachedModelProvider,
    ModelManager,
    ModelConfig,
    ModelType,
    GenerationRequest,
    GenerationResponse
)

from .feedback import (
    FeedbackProcessor,
    SimpleFeedbackProcessor,
    FeedbackData,
    FeedbackAnalysis,
    FeedbackSource,
    FeedbackType as FeedbackTypeEnum
)

from .adaptation import (
    AdaptiveLearner,
    ExperienceBasedLearner,
    AdaptationConfig,
    AdaptationStrategy,
    AdaptationTarget,
    AdaptationAction,
    AdaptationResult
)

__all__ = [
    # Memory management
    "MemoryManager",
    "InMemoryManager",
    "RedisMemoryManager",
    "Experience",
    "Feedback",
    "FeedbackType",
    "RetrievalResult",

    # Model providers
    "ModelProvider",
    "DeepSeekProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "CachedModelProvider",
    "ModelManager",
    "ModelConfig",
    "ModelType",
    "GenerationRequest",
    "GenerationResponse",

    # Feedback processing
    "FeedbackProcessor",
    "SimpleFeedbackProcessor",
    "FeedbackData",
    "FeedbackAnalysis",
    "FeedbackSource",
    "FeedbackTypeEnum",

    # Adaptive learning
    "AdaptiveLearner",
    "ExperienceBasedLearner",
    "AdaptationConfig",
    "AdaptationStrategy",
    "AdaptationTarget",
    "AdaptationAction",
    "AdaptationResult"
]