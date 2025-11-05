"""
学习相关接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio

from ..models import (
    Experience,
    Feedback,
    GenerationResult,
    RetrievalResult,
    QueryResult,
    ModelType,
    FeedbackType,
)


class IMemoryManager(ABC):
    """记忆管理接口"""

    @abstractmethod
    async def store_experience(self, experience: Experience) -> bool:
        """存储经验"""
        pass

    @abstractmethod
    async def retrieve_similar_experiences(self, query: str,
                                        top_k: int = 3,
                                        filters: Optional[Dict[str, Any]] = None) -> List[Experience]:
        """检索相似经验"""
        pass

    @abstractmethod
    async def update_feedback(self, experience_id: str,
                            feedback: Feedback) -> bool:
        """更新反馈"""
        pass

    @abstractmethod
    async def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """根据ID获取经验"""
        pass

    @abstractmethod
    async def delete_experience(self, experience_id: str) -> bool:
        """删除经验"""
        pass

    @abstractmethod
    async def search_experiences(self, query: str,
                               filters: Optional[Dict[str, Any]] = None,
                               limit: int = 100) -> List[Experience]:
        """搜索经验"""
        pass

    @abstractmethod
    async def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        pass

    @abstractmethod
    async def cleanup_old_experiences(self, days: int = 30) -> int:
        """清理旧经验"""
        pass


class IModelProvider(ABC):
    """模型提供者接口"""

    @abstractmethod
    async def generate(self, prompt: str,
                      context: Dict[str, Any]) -> GenerationResult:
        """生成响应"""
        pass

    @abstractmethod
    async def stream_generate(self, prompt: str,
                            context: Dict[str, Any]) -> AsyncIterator[str]:
        """流式生成"""
        pass

    @abstractmethod
    def should_use_for_task(self, task_type: str,
                          complexity: float,
                          context: Dict[str, Any]) -> bool:
        """是否应该用于此任务"""
        pass

    @abstractmethod
    def estimate_cost(self, prompt: str, response: str) -> float:
        """估算成本"""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """估算Token数量"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass

    @abstractmethod
    def get_model_type(self) -> ModelType:
        """获取模型类型"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


class IAdaptiveLearner(ABC):
    """自适应学习接口"""

    @abstractmethod
    async def learn_from_feedback(self, feedback_list: List[Feedback]) -> bool:
        """从反馈中学习"""
        pass

    @abstractmethod
    async def update_strategy(self, performance_metrics: Dict[str, float]) -> bool:
        """更新策略"""
        pass

    @abstractmethod
    async def should_trigger_learning(self, context: Dict[str, Any]) -> bool:
        """是否触发学习"""
        pass

    @abstractmethod
    async def get_learning_progress(self) -> Dict[str, Any]:
        """获取学习进度"""
        pass

    @abstractmethod
    async def reset_learning_state(self) -> bool:
        """重置学习状态"""
        pass

    @abstractmethod
    def get_learning_rate(self) -> float:
        """获取学习率"""
        pass

    @abstractmethod
    def set_learning_rate(self, rate: float) -> None:
        """设置学习率"""
        pass


class IFeedbackProcessor(ABC):
    """反馈处理器接口"""

    @abstractmethod
    async def process_feedback(self, feedback: Feedback) -> Dict[str, Any]:
        """处理反馈"""
        pass

    @abstractmethod
    async def batch_process_feedback(self, feedback_list: List[Feedback]) -> List[Dict[str, Any]]:
        """批量处理反馈"""
        pass

    @abstractmethod
    async def analyze_feedback_patterns(self, feedback_list: List[Feedback]) -> Dict[str, Any]:
        """分析反馈模式"""
        pass

    @abstractmethod
    async def get_feedback_summary(self, time_range: Optional[str] = None) -> Dict[str, Any]:
        """获取反馈摘要"""
        pass

    @abstractmethod
    def validate_feedback(self, feedback: Feedback) -> bool:
        """验证反馈"""
        pass


class IModelSelector(ABC):
    """模型选择器接口"""

    @abstractmethod
    async def select_model(self, query: str,
                         context: Dict[str, Any],
                         available_models: List[IModelProvider]) -> IModelProvider:
        """选择模型"""
        pass

    @abstractmethod
    async def rank_models(self, query: str,
                         context: Dict[str, Any],
                         models: List[IModelProvider]) -> List[IModelProvider]:
        """模型排序"""
        pass

    @abstractmethod
    def get_selection_criteria(self) -> Dict[str, float]:
        """获取选择标准"""
        pass

    @abstractmethod
    def update_selection_weights(self, weights: Dict[str, float]) -> None:
        """更新选择权重"""
        pass

    @abstractmethod
    async def get_model_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """获取模型性能统计"""
        pass


class IExperienceRanker(ABC):
    """经验排序器接口"""

    @abstractmethod
    async def rank_experiences(self, query: str,
                             experiences: List[Experience],
                             context: Optional[Dict[str, Any]] = None) -> List[Experience]:
        """对经验排序"""
        pass

    @abstractmethod
    async def calculate_relevance_score(self, query: str,
                                      experience: Experience) -> float:
        """计算相关性分数"""
        pass

    @abstractmethod
    async def calculate_quality_score(self, experience: Experience) -> float:
        """计算质量分数"""
        pass

    @abstractmethod
    async def calculate_freshness_score(self, experience: Experience) -> float:
        """计算新鲜度分数"""
        pass

    @abstractmethod
    def get_ranking_weights(self) -> Dict[str, float]:
        """获取排序权重"""
        pass

    @abstractmethod
    def set_ranking_weights(self, weights: Dict[str, float]) -> None:
        """设置排序权重"""
        pass


class IPromptBuilder(ABC):
    """提示构建器接口"""

    @abstractmethod
    async def build_prompt(self, query: str,
                         retrieval_result: RetrievalResult,
                         experiences: List[Experience],
                         context: Dict[str, Any]) -> str:
        """构建提示"""
        pass

    @abstractmethod
    async def build_system_prompt(self, context: Dict[str, Any]) -> str:
        """构建系统提示"""
        pass

    @abstractmethod
    async def build_few_shot_examples(self, experiences: List[Experience],
                                    max_examples: int = 3) -> str:
        """构建少样本示例"""
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """获取提示模板"""
        pass

    @abstractmethod
    def set_prompt_template(self, template: str) -> None:
        """设置提示模板"""
        pass

    @abstractmethod
    async def optimize_prompt_length(self, prompt: str,
                                   max_tokens: int) -> str:
        """优化提示长度"""
        pass


class ICostOptimizer(ABC):
    """成本优化器接口"""

    @abstractmethod
    async def optimize_model_usage(self, query: str,
                                 context: Dict[str, Any],
                                 budget_constraint: float) -> Dict[str, Any]:
        """优化模型使用"""
        pass

    @abstractmethod
    async def estimate_total_cost(self, query: str,
                                context: Dict[str, Any]) -> float:
        """估算总成本"""
        pass

    @abstractmethod
    async def suggest_cost_saving_measures(self, current_cost: float,
                                         target_cost: float) -> List[str]:
        """建议成本节约措施"""
        pass

    @abstractmethod
    def get_cost_tracking_stats(self) -> Dict[str, float]:
        """获取成本跟踪统计"""
        pass

    @abstractmethod
    async def set_budget_limit(self, limit: float) -> None:
        """设置预算限制"""
        pass


class IQualityAssurance(ABC):
    """质量保证接口"""

    @abstractmethod
    async def validate_response(self, query: str,
                               response: str,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """验证响应质量"""
        pass

    @abstractmethod
    async def detect_anomalies(self, response: str,
                             expected_patterns: List[str]) -> List[str]:
        """检测异常"""
        pass

    @abstractmethod
    async def check_coherence(self, response: str) -> float:
        """检查连贯性"""
        pass

    @abstractmethod
    async def check_relevance(self, query: str, response: str) -> float:
        """检查相关性"""
        pass

    @abstractmethod
    async def check_safety(self, response: str) -> Dict[str, Any]:
        """检查安全性"""
        pass

    @abstractmethod
    def get_quality_thresholds(self) -> Dict[str, float]:
        """获取质量阈值"""
        pass

    @abstractmethod
    def set_quality_thresholds(self, thresholds: Dict[str, float]) -> None:
        """设置质量阈值"""
        pass