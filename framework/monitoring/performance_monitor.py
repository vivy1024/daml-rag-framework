# -*- coding: utf-8 -*-
"""
性能监控系统 v1.0 - DAG执行性能监控和统计

提供全面的性能监控和统计功能，包括：
1. DAG执行性能监控
2. LLM调用统计
3. 缓存命中率统计
4. 工具执行时间分析
5. 性能趋势分析

作者: BUILD_BODY Team
版本: v1.0.0
日期: 2025-12-12
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionMetrics:
    """工具执行指标"""
    tool_name: str
    execution_id: str
    user_id: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGExecutionMetrics:
    """DAG执行指标"""
    execution_id: str
    template_id: str
    template_name: str
    user_id: str
    start_time: float
    end_time: float
    total_duration: float
    tools_executed: int
    tools_succeeded: int
    tools_failed: int
    tools_cached: int
    parallel_groups: int
    max_parallel_degree: int
    cache_hit_rate: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallMetrics:
    """LLM调用指标"""
    call_id: str
    call_type: str  # "decision" or "analysis"
    model_name: str
    user_id: str
    start_time: float
    end_time: float
    duration: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    error_message: Optional[str] = None
    confidence: float = 0.0
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """缓存指标"""
    timestamp: float
    hits: int
    misses: int
    hit_rate: float
    preloads: int
    evictions: int
    total_size: int
    memory_cache_size: int
    tool_specific_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: float
    period_seconds: int
    dag_executions: int
    avg_dag_duration: float
    llm_calls: int
    avg_llm_duration: float
    cache_hit_rate: float
    success_rate: float
    top_tools: List[Tuple[str, int]]
    top_errors: List[Tuple[str, int]]


class PerformanceMonitor:
    """性能监控系统"""
    
    def __init__(self, max_history_size: int = 1000):
        """
        初始化性能监控系统
        
        Args:
            max_history_size: 最大历史记录数量
        """
        self.max_history_size = max_history_size
        
        # 工具执行历史
        self.tool_executions: deque = deque(maxlen=max_history_size)
        
        # DAG执行历史
        self.dag_executions: deque = deque(maxlen=max_history_size)
        
        # LLM调用历史
        self.llm_calls: deque = deque(maxlen=max_history_size)
        
        # 缓存指标历史
        self.cache_metrics_history: deque = deque(maxlen=100)
        
        # 实时统计
        self.tool_stats = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        })
        
        self.llm_stats = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration": 0.0,
            "total_tokens": 0,
            "fallback_count": 0
        })
        
        # 错误统计
        self.error_counts = defaultdict(int)
        
        logger.info("✅ 性能监控系统初始化完成")
    
    # ========== 工具执行监控 ==========
    
    def record_tool_execution(self, metrics: ToolExecutionMetrics):
        """
        记录工具执行指标
        
        Args:
            metrics: 工具执行指标
        """
        # 添加到历史记录
        self.tool_executions.append(metrics)
        
        # 更新实时统计
        stats = self.tool_stats[metrics.tool_name]
        stats["total_calls"] += 1
        
        if metrics.success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
            if metrics.error_message:
                self.error_counts[metrics.error_message] += 1
        
        stats["total_duration"] += metrics.duration
        
        if metrics.cache_hit:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1
        
        logger.debug(
            f"📊 记录工具执行: {metrics.tool_name}, "
            f"耗时: {metrics.duration:.2f}s, "
            f"成功: {metrics.success}"
        )
    
    def get_tool_statistics(
        self,
        tool_name: Optional[str] = None,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取工具统计信息
        
        Args:
            tool_name: 工具名称（可选，不指定则返回所有工具）
            time_window_seconds: 时间窗口（秒）
            
        Returns:
            Dict[str, Any]: 工具统计信息
        """
        # 过滤时间窗口
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            filtered_executions = [
                m for m in self.tool_executions
                if m.start_time >= cutoff_time
            ]
        else:
            filtered_executions = list(self.tool_executions)
        
        # 过滤工具名称
        if tool_name:
            filtered_executions = [
                m for m in filtered_executions
                if m.tool_name == tool_name
            ]
        
        if not filtered_executions:
            return {
                "tool_name": tool_name,
                "total_calls": 0,
                "message": "没有找到匹配的执行记录"
            }
        
        # 计算统计
        total_calls = len(filtered_executions)
        successful_calls = sum(1 for m in filtered_executions if m.success)
        failed_calls = total_calls - successful_calls
        cache_hits = sum(1 for m in filtered_executions if m.cache_hit)
        
        durations = [m.duration for m in filtered_executions]
        
        return {
            "tool_name": tool_name or "all",
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "cache_hits": cache_hits,
            "cache_misses": total_calls - cache_hits,
            "cache_hit_rate": cache_hits / total_calls if total_calls > 0 else 0.0,
            "duration_stats": {
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0,
                "avg": statistics.mean(durations) if durations else 0.0,
                "median": statistics.median(durations) if durations else 0.0,
                "p95": self._calculate_percentile(durations, 0.95) if durations else 0.0,
                "p99": self._calculate_percentile(durations, 0.99) if durations else 0.0
            },
            "time_window_seconds": time_window_seconds
        }
    
    # ========== DAG执行监控 ==========
    
    def record_dag_execution(self, metrics: DAGExecutionMetrics):
        """
        记录DAG执行指标
        
        Args:
            metrics: DAG执行指标
        """
        self.dag_executions.append(metrics)
        
        logger.debug(
            f"📊 记录DAG执行: {metrics.template_name}, "
            f"耗时: {metrics.total_duration:.2f}s, "
            f"成功率: {metrics.success_rate:.1%}"
        )
    
    def get_dag_statistics(
        self,
        template_id: Optional[str] = None,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取DAG统计信息
        
        Args:
            template_id: 模板ID（可选）
            time_window_seconds: 时间窗口（秒）
            
        Returns:
            Dict[str, Any]: DAG统计信息
        """
        # 过滤时间窗口
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            filtered_executions = [
                m for m in self.dag_executions
                if m.start_time >= cutoff_time
            ]
        else:
            filtered_executions = list(self.dag_executions)
        
        # 过滤模板ID
        if template_id:
            filtered_executions = [
                m for m in filtered_executions
                if m.template_id == template_id
            ]
        
        if not filtered_executions:
            return {
                "template_id": template_id,
                "total_executions": 0,
                "message": "没有找到匹配的执行记录"
            }
        
        # 计算统计
        total_executions = len(filtered_executions)
        durations = [m.total_duration for m in filtered_executions]
        success_rates = [m.success_rate for m in filtered_executions]
        cache_hit_rates = [m.cache_hit_rate for m in filtered_executions]
        
        return {
            "template_id": template_id or "all",
            "total_executions": total_executions,
            "duration_stats": {
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0,
                "avg": statistics.mean(durations) if durations else 0.0,
                "median": statistics.median(durations) if durations else 0.0,
                "p95": self._calculate_percentile(durations, 0.95) if durations else 0.0
            },
            "success_rate": {
                "avg": statistics.mean(success_rates) if success_rates else 0.0,
                "min": min(success_rates) if success_rates else 0.0,
                "max": max(success_rates) if success_rates else 0.0
            },
            "cache_hit_rate": {
                "avg": statistics.mean(cache_hit_rates) if cache_hit_rates else 0.0,
                "min": min(cache_hit_rates) if cache_hit_rates else 0.0,
                "max": max(cache_hit_rates) if cache_hit_rates else 0.0
            },
            "avg_tools_executed": statistics.mean([m.tools_executed for m in filtered_executions]) if filtered_executions else 0.0,
            "avg_parallel_groups": statistics.mean([m.parallel_groups for m in filtered_executions]) if filtered_executions else 0.0,
            "time_window_seconds": time_window_seconds
        }
    
    # ========== LLM调用监控 ==========
    
    def record_llm_call(self, metrics: LLMCallMetrics):
        """
        记录LLM调用指标
        
        Args:
            metrics: LLM调用指标
        """
        self.llm_calls.append(metrics)
        
        # 更新实时统计
        stats = self.llm_stats[metrics.call_type]
        stats["total_calls"] += 1
        
        if metrics.success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        stats["total_duration"] += metrics.duration
        stats["total_tokens"] += metrics.total_tokens
        
        if metrics.fallback_used:
            stats["fallback_count"] += 1
        
        logger.debug(
            f"📊 记录LLM调用: {metrics.call_type}, "
            f"耗时: {metrics.duration:.2f}s, "
            f"tokens: {metrics.total_tokens}"
        )
    
    def get_llm_statistics(
        self,
        call_type: Optional[str] = None,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取LLM统计信息
        
        Args:
            call_type: 调用类型（可选）
            time_window_seconds: 时间窗口（秒）
            
        Returns:
            Dict[str, Any]: LLM统计信息
        """
        # 过滤时间窗口
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            filtered_calls = [
                m for m in self.llm_calls
                if m.start_time >= cutoff_time
            ]
        else:
            filtered_calls = list(self.llm_calls)
        
        # 过滤调用类型
        if call_type:
            filtered_calls = [
                m for m in filtered_calls
                if m.call_type == call_type
            ]
        
        if not filtered_calls:
            return {
                "call_type": call_type,
                "total_calls": 0,
                "message": "没有找到匹配的调用记录"
            }
        
        # 计算统计
        total_calls = len(filtered_calls)
        successful_calls = sum(1 for m in filtered_calls if m.success)
        fallback_calls = sum(1 for m in filtered_calls if m.fallback_used)
        
        durations = [m.duration for m in filtered_calls]
        total_tokens = sum(m.total_tokens for m in filtered_calls)
        prompt_tokens = sum(m.prompt_tokens for m in filtered_calls)
        completion_tokens = sum(m.completion_tokens for m in filtered_calls)
        
        return {
            "call_type": call_type or "all",
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "fallback_calls": fallback_calls,
            "fallback_rate": fallback_calls / total_calls if total_calls > 0 else 0.0,
            "duration_stats": {
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0,
                "avg": statistics.mean(durations) if durations else 0.0,
                "median": statistics.median(durations) if durations else 0.0,
                "p95": self._calculate_percentile(durations, 0.95) if durations else 0.0
            },
            "token_stats": {
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "avg_tokens_per_call": total_tokens / total_calls if total_calls > 0 else 0.0
            },
            "time_window_seconds": time_window_seconds
        }
    
    # ========== 缓存监控 ==========
    
    def record_cache_metrics(self, metrics: CacheMetrics):
        """
        记录缓存指标
        
        Args:
            metrics: 缓存指标
        """
        self.cache_metrics_history.append(metrics)
        
        logger.debug(
            f"📊 记录缓存指标: 命中率={metrics.hit_rate:.1%}, "
            f"预加载={metrics.preloads}, 淘汰={metrics.evictions}"
        )
    
    def get_cache_statistics(
        self,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Args:
            time_window_seconds: 时间窗口（秒）
            
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        # 过滤时间窗口
        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            filtered_metrics = [
                m for m in self.cache_metrics_history
                if m.timestamp >= cutoff_time
            ]
        else:
            filtered_metrics = list(self.cache_metrics_history)
        
        if not filtered_metrics:
            return {
                "message": "没有缓存指标记录"
            }
        
        # 获取最新指标
        latest_metrics = filtered_metrics[-1]
        
        # 计算趋势
        hit_rates = [m.hit_rate for m in filtered_metrics]
        
        return {
            "current": {
                "hits": latest_metrics.hits,
                "misses": latest_metrics.misses,
                "hit_rate": latest_metrics.hit_rate,
                "preloads": latest_metrics.preloads,
                "evictions": latest_metrics.evictions,
                "total_size": latest_metrics.total_size,
                "memory_cache_size": latest_metrics.memory_cache_size
            },
            "trends": {
                "avg_hit_rate": statistics.mean(hit_rates) if hit_rates else 0.0,
                "min_hit_rate": min(hit_rates) if hit_rates else 0.0,
                "max_hit_rate": max(hit_rates) if hit_rates else 0.0,
                "total_preloads": sum(m.preloads for m in filtered_metrics),
                "total_evictions": sum(m.evictions for m in filtered_metrics)
            },
            "tool_specific": latest_metrics.tool_specific_stats,
            "time_window_seconds": time_window_seconds
        }
    
    # ========== 综合统计和分析 ==========
    
    def get_performance_snapshot(
        self,
        time_window_seconds: int = 3600
    ) -> PerformanceSnapshot:
        """
        获取性能快照
        
        Args:
            time_window_seconds: 时间窗口（秒），默认1小时
            
        Returns:
            PerformanceSnapshot: 性能快照
        """
        cutoff_time = time.time() - time_window_seconds
        
        # 过滤DAG执行
        recent_dag_executions = [
            m for m in self.dag_executions
            if m.start_time >= cutoff_time
        ]
        
        # 过滤LLM调用
        recent_llm_calls = [
            m for m in self.llm_calls
            if m.start_time >= cutoff_time
        ]
        
        # 过滤工具执行
        recent_tool_executions = [
            m for m in self.tool_executions
            if m.start_time >= cutoff_time
        ]
        
        # 计算平均DAG执行时间
        avg_dag_duration = (
            statistics.mean([m.total_duration for m in recent_dag_executions])
            if recent_dag_executions else 0.0
        )
        
        # 计算平均LLM调用时间
        avg_llm_duration = (
            statistics.mean([m.duration for m in recent_llm_calls])
            if recent_llm_calls else 0.0
        )
        
        # 计算缓存命中率
        cache_hits = sum(1 for m in recent_tool_executions if m.cache_hit)
        total_tool_calls = len(recent_tool_executions)
        cache_hit_rate = cache_hits / total_tool_calls if total_tool_calls > 0 else 0.0
        
        # 计算成功率
        successful_dag = sum(1 for m in recent_dag_executions if m.success_rate > 0.9)
        success_rate = successful_dag / len(recent_dag_executions) if recent_dag_executions else 0.0
        
        # 统计最常用工具
        tool_counts = defaultdict(int)
        for m in recent_tool_executions:
            tool_counts[m.tool_name] += 1
        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 统计最常见错误
        error_counts_recent = defaultdict(int)
        for m in recent_tool_executions:
            if not m.success and m.error_message:
                error_counts_recent[m.error_message] += 1
        top_errors = sorted(error_counts_recent.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return PerformanceSnapshot(
            timestamp=time.time(),
            period_seconds=time_window_seconds,
            dag_executions=len(recent_dag_executions),
            avg_dag_duration=avg_dag_duration,
            llm_calls=len(recent_llm_calls),
            avg_llm_duration=avg_llm_duration,
            cache_hit_rate=cache_hit_rate,
            success_rate=success_rate,
            top_tools=top_tools,
            top_errors=top_errors
        )
    
    def get_performance_trends(
        self,
        time_window_seconds: int = 86400,  # 24小时
        interval_seconds: int = 3600  # 1小时间隔
    ) -> Dict[str, Any]:
        """
        获取性能趋势分析
        
        Args:
            time_window_seconds: 时间窗口（秒）
            interval_seconds: 间隔（秒）
            
        Returns:
            Dict[str, Any]: 性能趋势数据
        """
        current_time = time.time()
        start_time = current_time - time_window_seconds
        
        # 生成时间点
        time_points = []
        t = start_time
        while t <= current_time:
            time_points.append(t)
            t += interval_seconds
        
        # 为每个时间点计算指标
        trends = {
            "time_points": [datetime.fromtimestamp(t).isoformat() for t in time_points],
            "dag_execution_counts": [],
            "avg_dag_durations": [],
            "llm_call_counts": [],
            "avg_llm_durations": [],
            "cache_hit_rates": [],
            "success_rates": []
        }
        
        for i in range(len(time_points) - 1):
            interval_start = time_points[i]
            interval_end = time_points[i + 1]
            
            # 过滤该时间段的数据
            interval_dag_executions = [
                m for m in self.dag_executions
                if interval_start <= m.start_time < interval_end
            ]
            
            interval_llm_calls = [
                m for m in self.llm_calls
                if interval_start <= m.start_time < interval_end
            ]
            
            interval_tool_executions = [
                m for m in self.tool_executions
                if interval_start <= m.start_time < interval_end
            ]
            
            # 计算指标
            trends["dag_execution_counts"].append(len(interval_dag_executions))
            trends["avg_dag_durations"].append(
                statistics.mean([m.total_duration for m in interval_dag_executions])
                if interval_dag_executions else 0.0
            )
            
            trends["llm_call_counts"].append(len(interval_llm_calls))
            trends["avg_llm_durations"].append(
                statistics.mean([m.duration for m in interval_llm_calls])
                if interval_llm_calls else 0.0
            )
            
            cache_hits = sum(1 for m in interval_tool_executions if m.cache_hit)
            total_calls = len(interval_tool_executions)
            trends["cache_hit_rates"].append(
                cache_hits / total_calls if total_calls > 0 else 0.0
            )
            
            successful = sum(1 for m in interval_dag_executions if m.success_rate > 0.9)
            trends["success_rates"].append(
                successful / len(interval_dag_executions)
                if interval_dag_executions else 0.0
            )
        
        return trends
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        获取性能优化建议
        
        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        recommendations = []
        
        # 分析工具性能
        for tool_name, stats in self.tool_stats.items():
            if stats["total_calls"] < 10:
                continue
            
            avg_duration = stats["total_duration"] / stats["total_calls"]
            cache_hit_rate = (
                stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
                if (stats["cache_hits"] + stats["cache_misses"]) > 0 else 0.0
            )
            
            # 慢工具建议
            if avg_duration > 2.0:
                recommendations.append({
                    "type": "slow_tool",
                    "severity": "high",
                    "tool_name": tool_name,
                    "avg_duration": avg_duration,
                    "recommendation": f"工具 {tool_name} 平均执行时间 {avg_duration:.2f}s，建议优化或增加缓存"
                })
            
            # 低缓存命中率建议
            if cache_hit_rate < 0.3 and stats["total_calls"] > 20:
                recommendations.append({
                    "type": "low_cache_hit_rate",
                    "severity": "medium",
                    "tool_name": tool_name,
                    "cache_hit_rate": cache_hit_rate,
                    "recommendation": f"工具 {tool_name} 缓存命中率仅 {cache_hit_rate:.1%}，建议增加缓存TTL或预加载"
                })
            
            # 高失败率建议
            failure_rate = stats["failed_calls"] / stats["total_calls"]
            if failure_rate > 0.1:
                recommendations.append({
                    "type": "high_failure_rate",
                    "severity": "high",
                    "tool_name": tool_name,
                    "failure_rate": failure_rate,
                    "recommendation": f"工具 {tool_name} 失败率 {failure_rate:.1%}，建议检查错误日志并修复"
                })
        
        # 分析LLM性能
        for call_type, stats in self.llm_stats.items():
            if stats["total_calls"] < 5:
                continue
            
            avg_duration = stats["total_duration"] / stats["total_calls"]
            fallback_rate = stats["fallback_count"] / stats["total_calls"]
            
            # LLM慢调用建议
            if avg_duration > 5.0:
                recommendations.append({
                    "type": "slow_llm_call",
                    "severity": "medium",
                    "call_type": call_type,
                    "avg_duration": avg_duration,
                    "recommendation": f"LLM {call_type} 调用平均耗时 {avg_duration:.2f}s，建议优化提示词或使用更快的模型"
                })
            
            # 高降级率建议
            if fallback_rate > 0.2:
                recommendations.append({
                    "type": "high_fallback_rate",
                    "severity": "high",
                    "call_type": call_type,
                    "fallback_rate": fallback_rate,
                    "recommendation": f"LLM {call_type} 降级率 {fallback_rate:.1%}，建议检查LLM服务稳定性"
                })
        
        # 按严重程度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        return recommendations
    
    def export_metrics(
        self,
        format: str = "json",
        time_window_seconds: Optional[int] = None
    ) -> str:
        """
        导出性能指标
        
        Args:
            format: 导出格式（json或csv）
            time_window_seconds: 时间窗口（秒）
            
        Returns:
            str: 导出的数据
        """
        # 收集所有统计数据
        data = {
            "export_time": datetime.now().isoformat(),
            "time_window_seconds": time_window_seconds,
            "tool_statistics": self.get_tool_statistics(time_window_seconds=time_window_seconds),
            "dag_statistics": self.get_dag_statistics(time_window_seconds=time_window_seconds),
            "llm_statistics": self.get_llm_statistics(time_window_seconds=time_window_seconds),
            "cache_statistics": self.get_cache_statistics(time_window_seconds=time_window_seconds),
            "performance_snapshot": asdict(self.get_performance_snapshot(time_window_seconds or 3600)),
            "optimization_recommendations": self.get_optimization_recommendations()
        }
        
        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        elif format == "csv":
            # 简化的CSV导出（仅包含关键指标）
            lines = ["metric,value"]
            lines.append(f"total_dag_executions,{len(self.dag_executions)}")
            lines.append(f"total_llm_calls,{len(self.llm_calls)}")
            lines.append(f"total_tool_executions,{len(self.tool_executions)}")
            return "\n".join(lines)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def reset_statistics(self):
        """重置所有统计数据"""
        self.tool_executions.clear()
        self.dag_executions.clear()
        self.llm_calls.clear()
        self.cache_metrics_history.clear()
        self.tool_stats.clear()
        self.llm_stats.clear()
        self.error_counts.clear()
        
        logger.info("🔄 性能统计已重置")
    
    # ========== 辅助方法 ==========
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """
        计算百分位数
        
        Args:
            data: 数据列表
            percentile: 百分位（0-1）
            
        Returns:
            float: 百分位值
        """
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]


# 全局性能监控实例
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控实例"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def set_performance_monitor(monitor: PerformanceMonitor):
    """设置全局性能监控实例"""
    global _global_performance_monitor
    _global_performance_monitor = monitor
