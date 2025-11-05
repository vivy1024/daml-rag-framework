"""
基础数据模型定义
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeedbackType(Enum):
    """反馈类型枚举"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ModelType(Enum):
    """模型类型枚举"""
    TEACHER = "teacher"
    STUDENT = "student"
    CUSTOM = "custom"


@dataclass
class Document:
    """文档模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Entity:
    """实体模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Relation:
    """关系模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: str = ""
    type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Path:
    """路径模型"""
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    score: float = 0.0

    @property
    def length(self) -> int:
        return len(self.entities)


@dataclass
class SimilarityResult:
    """相似度结果"""
    document: Document
    score: float
    explanation: Optional[str] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    documents: List[Document]
    entities: List[Entity] = field(default_factory=list)
    paths: List[Path] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    @property
    def sources(self) -> List[str]:
        """返回来源ID列表"""
        return [doc.id for doc in self.documents]

    @property
    def top_k_score(self) -> float:
        """返回最高分数"""
        return max(self.scores) if self.scores else 0.0


@dataclass
class Feedback:
    """反馈模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experience_id: str = ""
    user_id: str = ""
    feedback_type: FeedbackType = FeedbackType.NEUTRAL
    rating: float = 0.0
    comment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Experience:
    """经验模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    response: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    retrieval_result: Optional[RetrievalResult] = None
    model_used: str = ""
    feedback_list: List[Feedback] = field(default_factory=list)
    quality_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def add_feedback(self, feedback: Feedback):
        """添加反馈"""
        feedback.experience_id = self.id
        self.feedback_list.append(feedback)
        self._update_quality_score()

    def _update_quality_score(self):
        """更新质量分数"""
        if not self.feedback_list:
            return

        positive_count = sum(1 for f in self.feedback_list if f.feedback_type == FeedbackType.POSITIVE)
        total_count = len(self.feedback_list)
        self.success_rate = positive_count / total_count if total_count > 0 else 0.0

        # 计算加权平均分数
        total_score = sum(f.rating for f in self.feedback_list if f.rating > 0)
        rating_count = sum(1 for f in self.feedback_list if f.rating > 0)
        self.quality_score = total_score / rating_count if rating_count > 0 else 0.0


@dataclass
class ToolSchema:
    """工具模式"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """任务模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def add_dependency(self, task_id: str):
        """添加依赖"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def is_ready(self, completed_tasks: set) -> bool:
        """检查任务是否准备就绪"""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)


@dataclass
class DAG:
    """有向无环图"""
    nodes: Dict[str, Task] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)

    def add_task(self, task: Task):
        """添加任务"""
        self.nodes[task.id] = task
        if task.id not in self.edges:
            self.edges[task.id] = task.dependencies

    def topological_sort(self) -> List[List[str]]:
        """拓扑排序，返回层级列表"""
        in_degree = {task_id: len(deps) for task_id, deps in self.edges.items()}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current_level = queue[:]
            result.append(current_level)
            queue = []

            for task_id in current_level:
                for neighbor in self.edges.get(task_id, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # 检查是否有循环依赖
        if sum(len(level) for level in result) != len(self.nodes):
            raise ValueError("检测到循环依赖")

        return result


@dataclass
class Workflow:
    """工作流模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    dag: DAG = field(default_factory=DAG)
    context: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    results: Dict[str, ToolResult] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """生成结果"""
    content: str
    model_used: str
    token_count: int = 0
    cost: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    response: str
    sources: List[str] = field(default_factory=list)
    model_used: str = ""
    execution_time: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    experience_id: Optional[str] = None
    quality_score: float = 0.0

    def to_experience(self) -> Experience:
        """转换为经验对象"""
        experience = Experience(
            query=self.query,
            response=self.response,
            context=self.context,
            model_used=self.model_used,
            quality_score=self.quality_score
        )
        if self.experience_id:
            experience.id = self.experience_id
        return experience

    def add_source(self, source_id: str):
        """添加来源"""
        if source_id not in self.sources:
            self.sources.append(source_id)