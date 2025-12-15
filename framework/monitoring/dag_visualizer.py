# -*- coding: utf-8 -*-
"""
DAG可视化器 v1.0 - 工作流程可视化和调试

提供DAG结构的可视化表示、执行日志输出和调试模式支持。
帮助开发者理解执行过程和排查问题。

核心功能：
1. DAG结构可视化（ASCII图、Mermaid图）
2. 执行日志输出（详细的工具执行信息）
3. 调试模式（中间结果、决策过程）
4. 执行历史查询

作者: BUILD_BODY Team
版本: v1.0.0
日期: 2025-12-12
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class VisualizationFormat(Enum):
    """可视化格式"""
    ASCII = "ascii"           # ASCII艺术图
    MERMAID = "mermaid"       # Mermaid流程图
    JSON = "json"             # JSON结构
    TREE = "tree"             # 树形结构


class LogLevel(Enum):
    """日志级别"""
    MINIMAL = 1      # 最小日志（仅关键信息）
    NORMAL = 2       # 正常日志（标准执行信息）
    DETAILED = 3     # 详细日志（包含参数和结果）
    DEBUG = 4        # 调试日志（所有中间结果和决策过程）


@dataclass
class ExecutionLogEntry:
    """执行日志条目"""
    timestamp: float
    level: str
    tool_name: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DAGVisualizationResult:
    """DAG可视化结果"""
    format: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DAGVisualizer:
    """DAG可视化器"""
    
    def __init__(self, debug_mode: bool = False, log_level: LogLevel = LogLevel.NORMAL):
        """
        初始化DAG可视化器
        
        Args:
            debug_mode: 是否启用调试模式
            log_level: 日志级别
        """
        self.debug_mode = debug_mode
        self.log_level = log_level
        self.execution_logs: List[ExecutionLogEntry] = []
        
        logger.info(f"✅ DAG可视化器初始化完成 (调试模式: {debug_mode}, 日志级别: {log_level.name})")
    
    def visualize_dag_structure(
        self,
        template_name: str,
        tools: List[str],
        dependencies: Dict[str, List[str]],
        parallel_groups: Optional[List[List[str]]] = None,
        format: VisualizationFormat = VisualizationFormat.ASCII
    ) -> DAGVisualizationResult:
        """
        可视化DAG结构
        
        Args:
            template_name: 模板名称
            tools: 工具列表
            dependencies: 依赖关系
            parallel_groups: 并行组
            format: 可视化格式
            
        Returns:
            DAGVisualizationResult: 可视化结果
        """
        if format == VisualizationFormat.ASCII:
            content = self._generate_ascii_dag(template_name, tools, dependencies, parallel_groups)
        elif format == VisualizationFormat.MERMAID:
            content = self._generate_mermaid_dag(template_name, tools, dependencies)
        elif format == VisualizationFormat.JSON:
            content = self._generate_json_dag(template_name, tools, dependencies, parallel_groups)
        elif format == VisualizationFormat.TREE:
            content = self._generate_tree_dag(template_name, tools, dependencies)
        else:
            content = "不支持的可视化格式"
        
        return DAGVisualizationResult(
            format=format.value,
            content=content,
            metadata={
                "template_name": template_name,
                "total_tools": len(tools),
                "total_dependencies": sum(len(deps) for deps in dependencies.values()),
                "parallel_groups_count": len(parallel_groups) if parallel_groups else 0
            }
        )
    
    def _generate_ascii_dag(
        self,
        template_name: str,
        tools: List[str],
        dependencies: Dict[str, List[str]],
        parallel_groups: Optional[List[List[str]]] = None
    ) -> str:
        """生成ASCII艺术图"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"DAG结构可视化: {template_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # 计算层级
        levels = self._calculate_levels(tools, dependencies)
        
        # 按层级输出
        for level_num, level_tools in enumerate(levels):
            lines.append(f"层级 {level_num + 1}:")
            lines.append("─" * 40)
            
            # 检查是否是并行组
            is_parallel = self._is_parallel_level(level_tools, parallel_groups)
            
            if is_parallel:
                lines.append("  [并行执行]")
            
            for tool in level_tools:
                # 显示工具名称
                tool_display = f"  ┌─ {tool}"
                lines.append(tool_display)
                
                # 显示依赖
                deps = dependencies.get(tool, [])
                if deps:
                    lines.append(f"  │  依赖: {', '.join(deps)}")
                
                lines.append("  └─")
            
            lines.append("")
        
        # 统计信息
        lines.append("=" * 80)
        lines.append("统计信息:")
        lines.append(f"  总工具数: {len(tools)}")
        lines.append(f"  总层级数: {len(levels)}")
        lines.append(f"  总依赖数: {sum(len(deps) for deps in dependencies.values())}")
        if parallel_groups:
            lines.append(f"  并行组数: {len(parallel_groups)}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_mermaid_dag(
        self,
        template_name: str,
        tools: List[str],
        dependencies: Dict[str, List[str]]
    ) -> str:
        """生成Mermaid流程图"""
        lines = []
        lines.append("```mermaid")
        lines.append("graph TD")
        lines.append(f"    Start([开始: {template_name}])")
        lines.append("")
        
        # 添加节点
        for tool in tools:
            # 转换工具名为合法的节点ID
            node_id = tool.replace("-", "_").replace(" ", "_")
            lines.append(f"    {node_id}[{tool}]")
        
        lines.append("")
        
        # 添加起始连接
        first_level_tools = [tool for tool in tools if not dependencies.get(tool, [])]
        for tool in first_level_tools:
            node_id = tool.replace("-", "_").replace(" ", "_")
            lines.append(f"    Start --> {node_id}")
        
        # 添加依赖关系
        for tool, deps in dependencies.items():
            tool_id = tool.replace("-", "_").replace(" ", "_")
            for dep in deps:
                dep_id = dep.replace("-", "_").replace(" ", "_")
                lines.append(f"    {dep_id} --> {tool_id}")
        
        # 添加结束节点
        last_level_tools = self._find_last_level_tools(tools, dependencies)
        lines.append("")
        lines.append("    End([结束])")
        for tool in last_level_tools:
            node_id = tool.replace("-", "_").replace(" ", "_")
            lines.append(f"    {node_id} --> End")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def _generate_json_dag(
        self,
        template_name: str,
        tools: List[str],
        dependencies: Dict[str, List[str]],
        parallel_groups: Optional[List[List[str]]] = None
    ) -> str:
        """生成JSON结构"""
        dag_structure = {
            "template_name": template_name,
            "tools": tools,
            "dependencies": dependencies,
            "parallel_groups": parallel_groups or [],
            "levels": self._calculate_levels(tools, dependencies),
            "statistics": {
                "total_tools": len(tools),
                "total_dependencies": sum(len(deps) for deps in dependencies.values()),
                "total_levels": len(self._calculate_levels(tools, dependencies)),
                "parallel_groups_count": len(parallel_groups) if parallel_groups else 0
            }
        }
        
        return json.dumps(dag_structure, indent=2, ensure_ascii=False)
    
    def _generate_tree_dag(
        self,
        template_name: str,
        tools: List[str],
        dependencies: Dict[str, List[str]]
    ) -> str:
        """生成树形结构"""
        lines = []
        lines.append(f"{template_name}")
        lines.append("│")
        
        # 找到根节点（没有依赖的工具）
        root_tools = [tool for tool in tools if not dependencies.get(tool, [])]
        
        # 递归构建树
        visited = set()
        for i, root in enumerate(root_tools):
            is_last = (i == len(root_tools) - 1)
            self._build_tree_recursive(root, dependencies, tools, lines, "", is_last, visited)
        
        return "\n".join(lines)
    
    def _build_tree_recursive(
        self,
        tool: str,
        dependencies: Dict[str, List[str]],
        all_tools: List[str],
        lines: List[str],
        prefix: str,
        is_last: bool,
        visited: Set[str]
    ):
        """递归构建树形结构"""
        if tool in visited:
            return
        visited.add(tool)
        
        # 当前节点
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{tool}")
        
        # 找到依赖当前工具的子工具
        children = [t for t in all_tools if tool in dependencies.get(t, [])]
        
        # 递归处理子节点
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            self._build_tree_recursive(child, dependencies, all_tools, lines, new_prefix, child_is_last, visited)
    
    def _calculate_levels(
        self,
        tools: List[str],
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """计算DAG层级"""
        levels = []
        remaining_tools = set(tools)
        processed_tools = set()
        
        while remaining_tools:
            # 找到当前可以执行的工具（依赖都已处理）
            current_level = []
            for tool in remaining_tools:
                deps = dependencies.get(tool, [])
                if all(dep in processed_tools for dep in deps):
                    current_level.append(tool)
            
            if not current_level:
                # 检测到循环依赖
                logger.warning(f"⚠️ 检测到循环依赖，剩余工具: {remaining_tools}")
                break
            
            levels.append(current_level)
            remaining_tools -= set(current_level)
            processed_tools.update(current_level)
        
        return levels
    
    def _is_parallel_level(
        self,
        level_tools: List[str],
        parallel_groups: Optional[List[List[str]]] = None
    ) -> bool:
        """判断层级是否可并行执行"""
        if not parallel_groups:
            return len(level_tools) > 1
        
        # 检查是否在并行组中
        for group in parallel_groups:
            if set(level_tools).issubset(set(group)):
                return True
        
        return False
    
    def _find_last_level_tools(
        self,
        tools: List[str],
        dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """找到最后一层的工具（没有其他工具依赖它们）"""
        depended_tools = set()
        for deps in dependencies.values():
            depended_tools.update(deps)
        
        return [tool for tool in tools if tool not in depended_tools]
    
    def log_tool_execution(
        self,
        tool_name: str,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        记录工具执行日志
        
        Args:
            tool_name: 工具名称
            status: 执行状态
            message: 日志消息
            details: 详细信息
            duration: 执行时长
            error: 错误信息
        """
        # 根据日志级别决定是否记录
        if self.log_level == LogLevel.MINIMAL and status not in ["failed", "completed"]:
            return
        
        entry = ExecutionLogEntry(
            timestamp=time.time(),
            level=self._get_log_level_for_status(status),
            tool_name=tool_name,
            status=status,
            message=message,
            details=details if self.log_level.value >= LogLevel.DETAILED.value else None,
            duration=duration,
            error=error
        )
        
        self.execution_logs.append(entry)
        
        # 输出日志
        self._print_log_entry(entry)
    
    def _get_log_level_for_status(self, status: str) -> str:
        """根据状态获取日志级别"""
        status_map = {
            "pending": "INFO",
            "running": "INFO",
            "completed": "INFO",
            "failed": "ERROR",
            "skipped": "WARNING",
            "cancelled": "WARNING"
        }
        return status_map.get(status, "INFO")
    
    def _print_log_entry(self, entry: ExecutionLogEntry):
        """打印日志条目"""
        timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]
        
        # 状态图标
        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
            "cancelled": "🚫"
        }
        icon = status_icons.get(entry.status, "📝")
        
        # 基本信息
        log_msg = f"[{timestamp_str}] {icon} {entry.tool_name}: {entry.message}"
        
        # 添加时长
        if entry.duration is not None:
            log_msg += f" ({entry.duration:.2f}s)"
        
        # 输出日志
        if entry.level == "ERROR":
            logger.error(log_msg)
            if entry.error:
                logger.error(f"  错误详情: {entry.error}")
        elif entry.level == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # 详细信息（调试模式或详细日志）
        if self.debug_mode or self.log_level.value >= LogLevel.DETAILED.value:
            if entry.details:
                logger.debug(f"  详细信息: {json.dumps(entry.details, indent=2, ensure_ascii=False)}")
    
    def log_decision_process(
        self,
        stage: str,
        decision: str,
        reason: str,
        alternatives: Optional[List[str]] = None,
        confidence: Optional[float] = None
    ):
        """
        记录决策过程（调试模式）
        
        Args:
            stage: 决策阶段
            decision: 决策结果
            reason: 决策理由
            alternatives: 备选方案
            confidence: 置信度
        """
        if not self.debug_mode:
            return
        
        logger.debug("=" * 60)
        logger.debug(f"🤔 决策阶段: {stage}")
        logger.debug(f"📋 决策结果: {decision}")
        logger.debug(f"💡 决策理由: {reason}")
        
        if confidence is not None:
            logger.debug(f"🎯 置信度: {confidence:.2f}")
        
        if alternatives:
            logger.debug(f"🔄 备选方案: {', '.join(alternatives)}")
        
        logger.debug("=" * 60)
    
    def log_intermediate_result(
        self,
        stage: str,
        result_type: str,
        result_data: Any
    ):
        """
        记录中间结果（调试模式）
        
        Args:
            stage: 执行阶段
            result_type: 结果类型
            result_data: 结果数据
        """
        if not self.debug_mode:
            return
        
        logger.debug(f"📊 中间结果 [{stage}] - {result_type}:")
        
        if isinstance(result_data, (dict, list)):
            logger.debug(json.dumps(result_data, indent=2, ensure_ascii=False))
        else:
            logger.debug(str(result_data))
    
    def get_execution_logs(
        self,
        tool_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ExecutionLogEntry]:
        """
        获取执行日志
        
        Args:
            tool_name: 过滤工具名称
            status: 过滤状态
            limit: 限制数量
            
        Returns:
            List[ExecutionLogEntry]: 日志列表
        """
        logs = self.execution_logs
        
        # 过滤
        if tool_name:
            logs = [log for log in logs if log.tool_name == tool_name]
        
        if status:
            logs = [log for log in logs if log.status == status]
        
        # 限制数量
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def generate_execution_summary(self) -> str:
        """生成执行摘要"""
        if not self.execution_logs:
            return "暂无执行日志"
        
        # 统计信息
        total_logs = len(self.execution_logs)
        status_counts = {}
        tool_counts = {}
        total_duration = 0.0
        
        for log in self.execution_logs:
            # 状态统计
            status_counts[log.status] = status_counts.get(log.status, 0) + 1
            
            # 工具统计
            tool_counts[log.tool_name] = tool_counts.get(log.tool_name, 0) + 1
            
            # 时长统计
            if log.duration:
                total_duration += log.duration
        
        # 生成摘要
        lines = []
        lines.append("=" * 80)
        lines.append("执行摘要")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"总日志数: {total_logs}")
        lines.append(f"总执行时长: {total_duration:.2f}s")
        lines.append("")
        lines.append("状态统计:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status}: {count}")
        lines.append("")
        lines.append("工具统计:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {tool}: {count}次")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def clear_logs(self):
        """清空日志"""
        self.execution_logs.clear()
        logger.info("🗑️ 执行日志已清空")
    
    def export_logs_to_file(self, filepath: str):
        """
        导出日志到文件
        
        Args:
            filepath: 文件路径
        """
        try:
            logs_data = []
            for log in self.execution_logs:
                logs_data.append({
                    "timestamp": log.timestamp,
                    "datetime": datetime.fromtimestamp(log.timestamp).isoformat(),
                    "level": log.level,
                    "tool_name": log.tool_name,
                    "status": log.status,
                    "message": log.message,
                    "details": log.details,
                    "duration": log.duration,
                    "error": log.error
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(logs_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 日志已导出到: {filepath}")
        
        except Exception as e:
            logger.error(f"❌ 导出日志失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = DAGVisualizer(debug_mode=True, log_level=LogLevel.DEBUG)
    
    # 示例DAG结构
    template_name = "完整训练计划"
    tools = [
        "get_user_profile",
        "contraindications_checker",
        "injury_risk_assessor",
        "intelligent_exercise_selector",
        "muscle_group_volume_calculator",
        "professional_program_designer"
    ]
    dependencies = {
        "get_user_profile": [],
        "contraindications_checker": ["get_user_profile"],
        "injury_risk_assessor": ["get_user_profile"],
        "intelligent_exercise_selector": ["contraindications_checker", "injury_risk_assessor"],
        "muscle_group_volume_calculator": ["get_user_profile"],
        "professional_program_designer": ["intelligent_exercise_selector", "muscle_group_volume_calculator"]
    }
    parallel_groups = [
        ["get_user_profile"],
        ["contraindications_checker", "injury_risk_assessor", "muscle_group_volume_calculator"],
        ["intelligent_exercise_selector"],
        ["professional_program_designer"]
    ]
    
    # 测试ASCII可视化
    print("\n" + "=" * 80)
    print("测试1: ASCII可视化")
    print("=" * 80)
    result = visualizer.visualize_dag_structure(
        template_name, tools, dependencies, parallel_groups, VisualizationFormat.ASCII
    )
    print(result.content)
    
    # 测试Mermaid可视化
    print("\n" + "=" * 80)
    print("测试2: Mermaid可视化")
    print("=" * 80)
    result = visualizer.visualize_dag_structure(
        template_name, tools, dependencies, parallel_groups, VisualizationFormat.MERMAID
    )
    print(result.content)
    
    # 测试执行日志
    print("\n" + "=" * 80)
    print("测试3: 执行日志")
    print("=" * 80)
    visualizer.log_tool_execution("get_user_profile", "running", "开始获取用户档案")
    visualizer.log_tool_execution("get_user_profile", "completed", "用户档案获取成功", duration=0.5)
    visualizer.log_tool_execution("contraindications_checker", "running", "开始检查禁忌动作")
    visualizer.log_tool_execution("contraindications_checker", "completed", "禁忌检查完成", duration=1.2)
    
    # 测试决策过程
    print("\n" + "=" * 80)
    print("测试4: 决策过程")
    print("=" * 80)
    visualizer.log_decision_process(
        stage="DAG选择",
        decision="complete_training_plan",
        reason="用户需要完整的增肌训练计划",
        alternatives=["nutrition_plan", "exercise_selection"],
        confidence=0.95
    )
    
    # 测试执行摘要
    print("\n" + "=" * 80)
    print("测试5: 执行摘要")
    print("=" * 80)
    print(visualizer.generate_execution_summary())
