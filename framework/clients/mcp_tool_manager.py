"""
MCP工具调用管理器

管理MCP工具的调用、参数验证、结果转换和错误处理。
提供统一的MCP工具调用接口，供MCP编排器使用。

主要特性:
1. 工具调用管理 - 统一的MCP工具调用接口
2. 参数验证 - 确保参数符合MCP协议规范
3. 结果转换 - 将MCP原始结果转换为标准格式
4. 错误处理 - 捕获异常、记录日志、提供降级方案
5. 工具映射 - 任务名称到MCP工具的映射配置
6. 错误统计 - 记录错误次数和类型，用于监控

使用场景:
- 被MCPOrchestrator使用，提供工具调用的统一接口
- 简化MCP工具的参数验证和结果转换
- 提供降级方案，确保系统稳定性

作者: BUILD_BODY Team
版本: v1.1.0
日期: 2025-12-14
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MCPToolCallResult:
    """MCP工具调用结果"""
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    tool_name: str
    execution_time_ms: float
    timestamp: datetime
    fallback_used: bool = False
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None


class MCPToolNotFoundError(Exception):
    """MCP工具不存在错误"""
    pass


class MCPToolCallError(Exception):
    """MCP工具调用失败错误"""
    pass


class MCPConnectionError(Exception):
    """MCP连接错误"""
    pass


class MCPTimeoutError(Exception):
    """MCP调用超时错误"""
    pass


class MCPParameterError(Exception):
    """MCP参数错误"""
    pass


@dataclass
class ErrorStatistics:
    """错误统计"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    fallback_calls: int = 0
    error_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_by_tool: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None


class MCPToolManager:
    """
    MCP工具调用管理器
    
    提供统一的MCP工具调用接口，处理参数验证、结果转换和错误恢复。
    供MCPOrchestrator使用，简化MCP工具调用流程。
    """

    def __init__(self, mcp_client=None):
        """
        初始化MCP工具管理器
        
        Args:
            mcp_client: MCP客户端实例（可选，如果不提供则需要外部传入server_name和tool_name）
        """
        self.mcp_client = mcp_client
        self.tool_mapping = self._load_tool_mapping()
        self.logger = logger
        self.error_stats = ErrorStatistics()

    def _load_tool_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        加载工具映射配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 任务名称到MCP工具的映射表
        """
        # 工具映射表：任务名称 -> MCP工具配置
        return {
            # 禁忌症检查
            "check_contraindications": {
                "server_name": "comprehensive-fitness-coach-stdio",
                "tool_name": "contraindications_checker",
                "description": "检查用户的健康禁忌症",
                "required_params": ["user_profile", "exercises"],
                "optional_params": ["strict_mode"],
                "param_schema": {
                    "user_profile": "dict",
                    "exercises": "list",
                    "strict_mode": "bool"
                }
            },
            
            # 受伤风险评估
            "assess_injury_risk": {
                "server_name": "comprehensive-fitness-coach-stdio",
                "tool_name": "injury_risk_assessor",
                "description": "评估动作的受伤风险",
                "required_params": ["user_profile", "exercises"],
                "optional_params": ["risk_threshold"],
                "param_schema": {
                    "user_profile": "dict",
                    "exercises": "list",
                    "risk_threshold": "float"
                }
            },
            
            # 替代动作查找
            "find_exercise_alternatives": {
                "server_name": "comprehensive-fitness-coach-stdio",
                "tool_name": "exercise_alternatives_finder",
                "description": "查找替代动作",
                "required_params": ["exercise_id", "user_profile"],
                "optional_params": ["max_alternatives"],
                "param_schema": {
                    "exercise_id": "str",
                    "user_profile": "dict",
                    "max_alternatives": "int"
                }
            },
            
            # 肌肉训练量计算
            "calculate_muscle_volume": {
                "server_name": "comprehensive-fitness-coach-stdio",
                "tool_name": "muscle_volume_calculator",
                "description": "计算肌肉训练量",
                "required_params": ["exercises", "user_profile"],
                "optional_params": ["time_period"],
                "param_schema": {
                    "exercises": "list",
                    "user_profile": "dict",
                    "time_period": "str"
                }
            },
            
            # 动作模式平衡
            "balance_movement_patterns": {
                "server_name": "comprehensive-fitness-coach-stdio",
                "tool_name": "movement_pattern_balancer",
                "description": "平衡动作模式",
                "required_params": ["exercises"],
                "optional_params": ["target_balance"],
                "param_schema": {
                    "exercises": "list",
                    "target_balance": "dict"
                }
            },
            
            # 运动营养优化
            "optimize_exercise_nutrition": {
                "server_name": "comprehensive-fitness-coach-stdio",
                "tool_name": "exercise_nutrition_optimizer",
                "description": "优化运动营养",
                "required_params": ["user_profile", "training_plan"],
                "optional_params": ["nutrition_goals"],
                "param_schema": {
                    "user_profile": "dict",
                    "training_plan": "dict",
                    "nutrition_goals": "dict"
                }
            }
        }

    async def call_tool(
        self,
        task_name: str,
        parameters: Dict[str, Any],
        mcp_client=None,
        context: Optional[Dict[str, Any]] = None
    ) -> MCPToolCallResult:
        """
        调用MCP工具（增强版错误处理）
        
        Args:
            task_name: 任务名称（如 "check_contraindications"）
            parameters: 工具参数
            mcp_client: MCP客户端实例（可选，如果初始化时未提供）
            context: 执行上下文（可选）
        
        Returns:
            MCPToolCallResult: 标准化的工具调用结果
        
        Raises:
            MCPToolNotFoundError: 工具不存在
        """
        start_time = datetime.now()
        self.error_stats.total_calls += 1
        
        # 使用传入的客户端或初始化时的客户端
        client = mcp_client or self.mcp_client
        
        try:
            # 检查工具是否存在
            if task_name not in self.tool_mapping:
                error_msg = (
                    f"未知的MCP工具任务: {task_name}\n"
                    f"已知任务: {list(self.tool_mapping.keys())}\n"
                    f"请在tool_mapping中添加该任务的配置"
                )
                self.logger.error(error_msg)
                self._record_error("MCPToolNotFoundError", task_name, error_msg)
                raise MCPToolNotFoundError(error_msg)
            
            tool_config = self.tool_mapping[task_name]
            server_name = tool_config["server_name"]
            tool_name = tool_config["tool_name"]
            
            self.logger.info(
                f"🔧 调用MCP工具: {task_name} -> "
                f"{server_name}/{tool_name}"
            )
            
            # 验证参数
            if not self._validate_parameters(task_name, parameters):
                error_msg = f"参数验证失败: {task_name}"
                self.logger.error(error_msg)
                self._record_error("MCPParameterError", task_name, error_msg)
                raise MCPParameterError(error_msg)
            
            # 调用MCP工具
            if client:
                raw_result = await client.call_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=parameters
                )
            else:
                # 如果没有客户端，返回模拟结果
                self.logger.warning(f"⚠️ 没有MCP客户端，返回模拟结果: {task_name}")
                raw_result = {
                    "success": True,
                    "data": {"message": "模拟结果（无MCP客户端）"},
                    "tool_name": tool_name
                }
            
            # 转换结果
            converted_result = self._convert_result(raw_result, tool_name)
            
            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"✅ MCP工具调用成功: {task_name} "
                f"(耗时: {execution_time:.2f}ms)"
            )
            
            self.error_stats.successful_calls += 1
            
            return MCPToolCallResult(
                success=True,
                data=converted_result,
                error=None,
                tool_name=tool_name,
                execution_time_ms=execution_time,
                timestamp=datetime.now(),
                fallback_used=False,
                error_type=None,
                stack_trace=None
            )
            
        except MCPToolNotFoundError:
            # 工具不存在，直接抛出（不提供降级）
            raise
            
        except MCPParameterError:
            # 参数错误，直接抛出（不提供降级）
            raise
            
        except asyncio.TimeoutError as e:
            # 超时错误
            return self._handle_timeout_error(task_name, parameters, e, start_time)
            
        except ConnectionError as e:
            # 连接错误
            return self._handle_connection_error(task_name, parameters, e, start_time)
            
        except Exception as e:
            # 其他错误，记录详细日志并提供降级方案
            return self._handle_general_error(task_name, parameters, e, start_time)

    def _handle_timeout_error(
        self,
        task_name: str,
        parameters: Dict[str, Any],
        error: Exception,
        start_time: datetime
    ) -> MCPToolCallResult:
        """
        处理超时错误
        
        Args:
            task_name: 任务名称
            parameters: 参数
            error: 异常对象
            start_time: 开始时间
        
        Returns:
            MCPToolCallResult: 包含降级结果的调用结果
        """
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        tool_config = self.tool_mapping.get(task_name, {})
        error_msg = f"MCP工具调用超时: {task_name}"
        stack_trace = traceback.format_exc()
        
        self.logger.error(
            f"⏱️ {error_msg}\n"
            f"   工具名: {tool_config.get('tool_name', 'unknown')}\n"
            f"   服务器: {tool_config.get('server_name', 'unknown')}\n"
            f"   参数: {parameters}\n"
            f"   超时时间: {execution_time:.2f}ms\n"
            f"   错误详情: {str(error)}\n"
            f"   堆栈跟踪:\n{stack_trace}"
        )
        
        self._record_error("MCPTimeoutError", task_name, error_msg)
        
        # 提供降级方案
        fallback_result = self._get_fallback_result(
            task_name, 
            parameters, 
            error_msg,
            "timeout"
        )
        
        return MCPToolCallResult(
            success=False,
            data=fallback_result,
            error=error_msg,
            tool_name=tool_config.get("tool_name", "unknown"),
            execution_time_ms=execution_time,
            timestamp=datetime.now(),
            fallback_used=True,
            error_type="MCPTimeoutError",
            stack_trace=stack_trace
        )

    def _handle_connection_error(
        self,
        task_name: str,
        parameters: Dict[str, Any],
        error: Exception,
        start_time: datetime
    ) -> MCPToolCallResult:
        """
        处理连接错误
        
        Args:
            task_name: 任务名称
            parameters: 参数
            error: 异常对象
            start_time: 开始时间
        
        Returns:
            MCPToolCallResult: 包含降级结果的调用结果
        """
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        tool_config = self.tool_mapping.get(task_name, {})
        error_msg = f"MCP服务连接失败: {task_name}"
        stack_trace = traceback.format_exc()
        
        self.logger.error(
            f"🔌 {error_msg}\n"
            f"   工具名: {tool_config.get('tool_name', 'unknown')}\n"
            f"   服务器: {tool_config.get('server_name', 'unknown')}\n"
            f"   参数: {parameters}\n"
            f"   错误详情: {str(error)}\n"
            f"   堆栈跟踪:\n{stack_trace}\n"
            f"   建议: 请检查MCP服务器是否正常运行"
        )
        
        self._record_error("MCPConnectionError", task_name, error_msg)
        
        # 提供降级方案
        fallback_result = self._get_fallback_result(
            task_name, 
            parameters, 
            error_msg,
            "connection"
        )
        
        return MCPToolCallResult(
            success=False,
            data=fallback_result,
            error=error_msg,
            tool_name=tool_config.get("tool_name", "unknown"),
            execution_time_ms=execution_time,
            timestamp=datetime.now(),
            fallback_used=True,
            error_type="MCPConnectionError",
            stack_trace=stack_trace
        )

    def _handle_general_error(
        self,
        task_name: str,
        parameters: Dict[str, Any],
        error: Exception,
        start_time: datetime
    ) -> MCPToolCallResult:
        """
        处理一般错误
        
        Args:
            task_name: 任务名称
            parameters: 参数
            error: 异常对象
            start_time: 开始时间
        
        Returns:
            MCPToolCallResult: 包含降级结果的调用结果
        """
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        tool_config = self.tool_mapping.get(task_name, {})
        error_type = type(error).__name__
        error_msg = str(error)
        stack_trace = traceback.format_exc()
        
        self.logger.error(
            f"❌ MCP工具调用失败: {task_name}\n"
            f"   工具名: {tool_config.get('tool_name', 'unknown')}\n"
            f"   服务器: {tool_config.get('server_name', 'unknown')}\n"
            f"   参数: {parameters}\n"
            f"   错误类型: {error_type}\n"
            f"   错误详情: {error_msg}\n"
            f"   堆栈跟踪:\n{stack_trace}"
        )
        
        self._record_error(error_type, task_name, error_msg)
        
        # 提供降级方案
        fallback_result = self._get_fallback_result(
            task_name, 
            parameters, 
            error_msg,
            "general"
        )
        
        return MCPToolCallResult(
            success=False,
            data=fallback_result,
            error=error_msg,
            tool_name=tool_config.get("tool_name", "unknown"),
            execution_time_ms=execution_time,
            timestamp=datetime.now(),
            fallback_used=True,
            error_type=error_type,
            stack_trace=stack_trace
        )

    def _record_error(self, error_type: str, task_name: str, error_msg: str):
        """
        记录错误统计
        
        Args:
            error_type: 错误类型
            task_name: 任务名称
            error_msg: 错误信息
        """
        self.error_stats.failed_calls += 1
        self.error_stats.error_by_type[error_type] += 1
        self.error_stats.error_by_tool[task_name] += 1
        self.error_stats.last_error_time = datetime.now()
        self.error_stats.last_error_message = error_msg

    def _validate_parameters(
        self,
        task_name: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        验证参数是否符合MCP协议规范
        
        Args:
            task_name: 任务名称
            parameters: 工具参数
        
        Returns:
            bool: 参数是否有效
        """
        try:
            tool_config = self.tool_mapping.get(task_name)
            if not tool_config:
                self.logger.error(f"工具配置不存在: {task_name}")
                return False
            
            # 检查必需参数
            required_params = tool_config.get("required_params", [])
            for param in required_params:
                if param not in parameters:
                    self.logger.error(
                        f"缺少必需参数: {param} (任务: {task_name})"
                    )
                    return False
            
            # 检查参数类型（基本验证）
            param_schema = tool_config.get("param_schema", {})
            for param_name, param_value in parameters.items():
                if param_name in param_schema:
                    expected_type = param_schema[param_name]
                    actual_type = type(param_value).__name__
                    
                    # 简单的类型检查
                    if expected_type == "dict" and not isinstance(param_value, dict):
                        self.logger.warning(
                            f"参数类型不匹配: {param_name} "
                            f"(期望: {expected_type}, 实际: {actual_type})"
                        )
                    elif expected_type == "list" and not isinstance(param_value, list):
                        self.logger.warning(
                            f"参数类型不匹配: {param_name} "
                            f"(期望: {expected_type}, 实际: {actual_type})"
                        )
            
            self.logger.debug(f"✅ 参数验证通过: {task_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"参数验证异常: {e}", exc_info=True)
            return False

    def _convert_result(
        self,
        raw_result: Any,
        tool_name: str
    ) -> Dict[str, Any]:
        """
        将MCP原始结果转换为标准格式
        
        Args:
            raw_result: MCP原始结果
            tool_name: 工具名称
        
        Returns:
            Dict[str, Any]: 标准化的结果字典
        """
        try:
            # 如果已经是字典格式，直接返回
            if isinstance(raw_result, dict):
                # 确保包含必需字段
                standardized = {
                    "success": raw_result.get("success", True),
                    "data": raw_result.get("data", raw_result),
                    "error": raw_result.get("error"),
                    "tool_name": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
                return standardized
            
            # 如果是其他类型，包装成字典
            return {
                "success": True,
                "data": raw_result,
                "error": None,
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"结果转换失败: {e}", exc_info=True)
            return {
                "success": False,
                "data": None,
                "error": f"结果转换失败: {str(e)}",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat()
            }

    def _get_fallback_result(
        self,
        task_name: str,
        parameters: Dict[str, Any],
        error: str,
        error_category: str = "general"
    ) -> Dict[str, Any]:
        """
        获取降级结果（增强版）
        
        Args:
            task_name: 任务名称
            parameters: 原始参数
            error: 错误信息
            error_category: 错误类别（timeout/connection/general）
        
        Returns:
            Dict[str, Any]: 降级结果
        """
        self.error_stats.fallback_calls += 1
        
        # 根据错误类别提供不同的降级建议
        fallback_messages = {
            "timeout": f"MCP工具 {task_name} 响应超时，请稍后重试或联系管理员",
            "connection": f"MCP服务 {task_name} 暂时无法连接，请检查服务状态",
            "general": f"MCP工具 {task_name} 暂时不可用，请稍后重试"
        }
        
        return {
            "success": False,
            "data": None,
            "error": error,
            "fallback": True,
            "error_category": error_category,
            "task_name": task_name,
            "message": fallback_messages.get(error_category, fallback_messages["general"]),
            "timestamp": datetime.now().isoformat(),
            "suggestion": self._get_error_suggestion(error_category)
        }

    def _get_error_suggestion(self, error_category: str) -> str:
        """
        获取错误建议
        
        Args:
            error_category: 错误类别
        
        Returns:
            str: 错误建议
        """
        suggestions = {
            "timeout": "建议：1) 检查网络连接 2) 增加超时时间 3) 检查MCP服务负载",
            "connection": "建议：1) 确认MCP服务正在运行 2) 检查服务端口 3) 查看服务日志",
            "general": "建议：1) 查看详细错误日志 2) 检查参数格式 3) 联系技术支持"
        }
        return suggestions.get(error_category, suggestions["general"])

    def get_tool_info(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具信息
        
        Args:
            task_name: 任务名称
        
        Returns:
            Optional[Dict[str, Any]]: 工具配置信息
        """
        return self.tool_mapping.get(task_name)

    def list_available_tools(self) -> List[str]:
        """
        列出所有可用的工具
        
        Returns:
            List[str]: 工具名称列表
        """
        return list(self.tool_mapping.keys())

    def get_tool_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        获取完整的工具映射表
        
        Returns:
            Dict[str, Dict[str, Any]]: 工具映射表
        """
        return self.tool_mapping.copy()

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        Returns:
            Dict[str, Any]: 错误统计
        """
        success_rate = (
            self.error_stats.successful_calls / self.error_stats.total_calls * 100
            if self.error_stats.total_calls > 0 else 0
        )
        
        return {
            "total_calls": self.error_stats.total_calls,
            "successful_calls": self.error_stats.successful_calls,
            "failed_calls": self.error_stats.failed_calls,
            "fallback_calls": self.error_stats.fallback_calls,
            "success_rate": f"{success_rate:.2f}%",
            "error_by_type": dict(self.error_stats.error_by_type),
            "error_by_tool": dict(self.error_stats.error_by_tool),
            "last_error_time": (
                self.error_stats.last_error_time.isoformat()
                if self.error_stats.last_error_time else None
            ),
            "last_error_message": self.error_stats.last_error_message
        }

    def reset_error_statistics(self):
        """重置错误统计"""
        self.error_stats = ErrorStatistics()
        self.logger.info("🔄 错误统计已重置")


# 便捷工厂函数
def create_mcp_tool_manager(mcp_client=None) -> MCPToolManager:
    """
    创建MCP工具管理器
    
    Args:
        mcp_client: MCP客户端实例（可选）
    
    Returns:
        MCPToolManager: MCP工具管理器实例
    """
    return MCPToolManager(mcp_client)
