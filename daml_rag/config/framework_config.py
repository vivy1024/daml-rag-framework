"""
框架配置定义
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
import json
import yaml


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 向量检索配置
    vector_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_dimension: int = 768
    vector_index_type: str = "faiss"
    top_k: int = 5
    similarity_threshold: float = 0.6

    # 缓存配置
    cache_ttl: int = 300  # 秒
    cache_max_size: int = 1000
    enable_cache: bool = True

    # 知识图谱配置
    enable_kg: bool = True
    kg_max_depth: int = 3
    kg_max_paths: int = 10

    # 规则过滤配置
    enable_rules: bool = True
    min_reward_threshold: float = 3.5
    max_results: int = 20

    # 批处理配置
    batch_size: int = 32
    max_batch_size: int = 128


@dataclass
class OrchestrationConfig:
    """编排配置"""
    # 任务执行配置
    max_parallel_tasks: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # 调度配置
    scheduling_strategy: str = "topological"  # topological, priority, resource_aware
    max_execution_time: int = 300  # 秒

    # 资源管理
    max_memory_usage: int = 4096  # MB
    max_cpu_usage: float = 0.8

    # MCP配置
    mcp_timeout: int = 10
    mcp_retry_attempts: int = 2

    # 缓存配置
    enable_workflow_cache: bool = True
    workflow_cache_ttl: int = 600

    # 监控配置
    enable_monitoring: bool = True
    metrics_interval: int = 60  # 秒


@dataclass
class LearningConfig:
    """学习配置"""
    # 模型配置
    teacher_model: str = "deepseek"
    student_model: str = "ollama-qwen2.5"
    model_fallback_enabled: bool = True

    # 经验管理配置
    experience_threshold: float = 3.5
    max_experiences_per_query: int = 3
    experience_retention_days: int = 90

    # 反馈学习配置
    feedback_weight: float = 0.8
    feedback_min_count: int = 3
    adaptive_threshold: float = 0.7

    # 学习策略配置
    learning_rate: float = 0.1
    learning_batch_size: int = 10
    learning_interval: int = 100  # 查询次数

    # 质量保证配置
    quality_threshold: float = 0.7
    quality_check_enabled: bool = True
    anomaly_detection_enabled: bool = True


@dataclass
class QualityConfig:
    """质量监控配置"""
    # 监控配置
    enable_monitoring: bool = True
    monitoring_interval: int = 60  # 秒

    # 异常检测配置
    anomaly_threshold: float = 0.8
    anomaly_window_size: int = 100

    # 信誉系统配置
    reputation_enabled: bool = True
    reputation_decay_rate: float = 0.95
    reputation_weight: float = 0.3

    # 质量阈值配置
    response_quality_threshold: float = 0.7
    coherence_threshold: float = 0.6
    relevance_threshold: float = 0.8

    # 报告配置
    enable_reports: bool = True
    report_interval: int = 3600  # 秒
    report_retention_days: int = 30


@dataclass
class DomainConfig:
    """领域配置"""
    # 基本信息
    domain_name: str = "fitness"
    domain_version: str = "1.0.0"
    description: str = ""

    # 知识图谱配置
    knowledge_graph_path: str = "./data/knowledge_graph.db"
    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)

    # MCP服务器配置
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)

    # 意图模式配置
    intent_patterns: List[str] = field(default_factory=list)

    # 领域特定配置
    domain_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """安全配置"""
    # API安全配置
    api_key_required: bool = False
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 秒

    # 内容安全配置
    content_filter_enabled: bool = True
    sensitive_content_detection: bool = True

    # 数据隐私配置
    data_encryption_enabled: bool = False
    anonymization_enabled: bool = True
    data_retention_days: int = 365


@dataclass
class LoggingConfig:
    """日志配置"""
    # 基本日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 文件日志配置
    log_to_file: bool = True
    log_file_path: str = "./logs/玉珍健身.log"
    log_file_rotation: bool = True
    log_max_file_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # 结构化日志配置
    structured_logging: bool = False
    log_to_json: bool = False

    # 特定模块日志配置
    component_log_levels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DAMLRAGConfig:
    """玉珍健身框架主配置"""
    # 核心配置
    domain: str = "fitness"
    debug: bool = False
    environment: str = "development"  # development, testing, production

    # 组件配置
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    domain_config: DomainConfig = field(default_factory=DomainConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # 系统配置
    max_concurrent_queries: int = 50
    query_timeout: int = 60
    health_check_interval: int = 30

    # 性能配置
    enable_performance_monitoring: bool = True
    metrics_collection_enabled: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> 'DAMLRAGConfig':
        """从文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        file_ext = os.path.splitext(config_path)[1].lower()

        with open(config_path, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif file_ext == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DAMLRAGConfig':
        """从字典创建配置"""
        # 处理嵌套配置对象
        retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
        orchestration_config = OrchestrationConfig(**config_dict.get('orchestration', {}))
        learning_config = LearningConfig(**config_dict.get('learning', {}))
        quality_config = QualityConfig(**config_dict.get('quality', {}))
        domain_config = DomainConfig(**config_dict.get('domain_config', {}))
        security_config = SecurityConfig(**config_dict.get('security', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))

        return cls(
            domain=config_dict.get('domain', 'fitness'),
            debug=config_dict.get('debug', False),
            environment=config_dict.get('environment', 'development'),
            retrieval=retrieval_config,
            orchestration=orchestration_config,
            learning=learning_config,
            quality=quality_config,
            domain_config=domain_config,
            security=security_config,
            logging=logging_config,
            max_concurrent_queries=config_dict.get('max_concurrent_queries', 50),
            query_timeout=config_dict.get('query_timeout', 60),
            health_check_interval=config_dict.get('health_check_interval', 30),
            enable_performance_monitoring=config_dict.get('enable_performance_monitoring', True),
            metrics_collection_enabled=config_dict.get('metrics_collection_enabled', True),
        )

    @classmethod
    def from_env(cls) -> 'DAMLRAGConfig':
        """从环境变量创建配置"""
        config_dict = {}

        # 从环境变量读取配置
        env_mapping = {
            'DAML_RAG_DOMAIN': ('domain', str),
            'DAML_RAG_DEBUG': ('debug', bool),
            'DAML_RAG_ENVIRONMENT': ('environment', str),
            'DAML_RAG_MAX_CONCURRENT_QUERIES': ('max_concurrent_queries', int),
            'DAML_RAG_QUERY_TIMEOUT': ('query_timeout', int),
        }

        for env_key, (config_key, value_type) in env_mapping.items():
            if env_key in os.environ:
                if value_type == bool:
                    config_dict[config_key] = os.environ[env_key].lower() in ['true', '1', 'yes', 'on']
                elif value_type == int:
                    config_dict[config_key] = int(os.environ[env_key])
                else:
                    config_dict[config_key] = os.environ[env_key]

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'domain': self.domain,
            'debug': self.debug,
            'environment': self.environment,
            'retrieval': self.retrieval.__dict__,
            'orchestration': self.orchestration.__dict__,
            'learning': self.learning.__dict__,
            'quality': self.quality.__dict__,
            'domain_config': self.domain_config.__dict__,
            'security': self.security.__dict__,
            'logging': self.logging.__dict__,
            'max_concurrent_queries': self.max_concurrent_queries,
            'query_timeout': self.query_timeout,
            'health_check_interval': self.health_check_interval,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'metrics_collection_enabled': self.metrics_collection_enabled,
        }

    def save_to_file(self, config_path: str) -> None:
        """保存到文件"""
        config_dict = self.to_dict()

        file_ext = os.path.splitext(config_path)[1].lower()

        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif file_ext == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")

    def validate(self) -> List[str]:
        """验证配置"""
        errors = []

        # 验证基本配置
        if not self.domain:
            errors.append("domain 不能为空")

        if self.max_concurrent_queries <= 0:
            errors.append("max_concurrent_queries 必须大于0")

        if self.query_timeout <= 0:
            errors.append("query_timeout 必须大于0")

        # 验证检索配置
        if self.retrieval.top_k <= 0:
            errors.append("retrieval.top_k 必须大于0")

        if not 0 <= self.retrieval.similarity_threshold <= 1:
            errors.append("retrieval.similarity_threshold 必须在0-1之间")

        # 验证编排配置
        if self.orchestration.max_parallel_tasks <= 0:
            errors.append("orchestration.max_parallel_tasks 必须大于0")

        if self.orchestration.timeout_seconds <= 0:
            errors.append("orchestration.timeout_seconds 必须大于0")

        # 验证学习配置
        if not 0 <= self.learning.experience_threshold <= 5:
            errors.append("learning.experience_threshold 必须在0-5之间")

        if not 0 <= self.learning.feedback_weight <= 1:
            errors.append("learning.feedback_weight 必须在0-1之间")

        return errors

    def merge_with(self, other: 'DAMLRAGConfig') -> 'DAMLRAGConfig':
        """与另一个配置合并"""
        current_dict = self.to_dict()
        other_dict = other.to_dict()

        def deep_merge(dict1: dict, dict2: dict) -> dict:
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_dict = deep_merge(current_dict, other_dict)
        return DAMLRAGConfig.from_dict(merged_dict)