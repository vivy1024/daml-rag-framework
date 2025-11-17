# -*- coding: utf-8 -*-
"""
DAML-RAG框架组件注册系统 v2.0

提供组件的自动发现、注册、依赖注入和生命周期管理。

版本：v2.0.0
更新日期：2025-11-17
设计原则：自动发现、依赖注入、生命周期管理
"""

import asyncio
import inspect
import logging
from typing import Dict, Any, List, Optional, Type, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import importlib.util
import sys

from ..interfaces.base import IComponent, IConfigurable, ComponentState
from ..interfaces.retrieval import IRetriever, IThreeLayerRetriever
from ..interfaces.orchestration import IOrchestrator, ITool, IToolRegistry
from ..interfaces.quality import IQualityChecker
from ..interfaces.storage import IStorage

logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """组件信息"""
    name: str
    component_class: Type[IComponent]
    instance: Optional[IComponent] = None
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    singleton: bool = True
    auto_discover: bool = True
    category: str = "general"
    priority: int = 0
    initialized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentCategory(Enum):
    """组件分类"""
    RETRIEVAL = "retrieval"
    ORCHESTRATION = "orchestration"
    QUALITY = "quality"
    STORAGE = "storage"
    UTILITY = "utility"
    DOMAIN = "domain"


class RegistryState(Enum):
    """注册器状态"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


class ComponentRegistry:
    """
    组件注册器

    负责组件的注册、发现、依赖注入和生命周期管理。
    """

    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._instances: Dict[str, IComponent] = {}
        self._state = RegistryState.UNINITIALIZED
        self._initialization_order: List[str] = []
        self._config: Dict[str, Any] = {}
        self._discovery_paths: List[str] = []
        self._event_handlers: Dict[str, List[Callable]] = {
            "component_registered": [],
            "component_unregistered": [],
            "component_initialized": [],
            "component_failed": []
        }

    def set_config(self, config: Dict[str, Any]) -> None:
        """设置注册器配置"""
        self._config.update(config)

    def get_config(self) -> Dict[str, Any]:
        """获取注册器配置"""
        return self._config.copy()

    def add_discovery_path(self, path: str) -> None:
        """添加组件发现路径"""
        if path not in self._discovery_paths:
            self._discovery_paths.append(path)

    async def register_component_class(
        self,
        component_class: Type[IComponent],
        name: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        singleton: bool = True,
        auto_discover: bool = True,
        category: str = "general",
        priority: int = 0,
        **metadata
    ) -> bool:
        """
        注册组件类

        Args:
            component_class: 组件类
            name: 组件名称（可选）
            dependencies: 依赖组件列表
            config: 默认配置
            singleton: 是否单例
            auto_discover: 是否自动发现
            category: 组件分类
            priority: 优先级
            **metadata: 其他元数据

        Returns:
            bool: 注册是否成功
        """
        try:
            if name is None:
                name = component_class.__name__

            # 验证组件类
            if not issubclass(component_class, IComponent):
                logger.error(f"组件类 {component_class} 必须实现 IComponent 接口")
                return False

            # 检查是否已注册
            if name in self._components:
                logger.warning(f"组件 {name} 已存在，将被覆盖")

            # 创建组件信息
            component_info = ComponentInfo(
                name=name,
                component_class=component_class,
                dependencies=dependencies or [],
                config=config or {},
                singleton=singleton,
                auto_discover=auto_discover,
                category=category,
                priority=priority,
                metadata=metadata
            )

            self._components[name] = component_info

            # 触发注册事件
            await self._emit_event("component_registered", component_info)

            logger.info(f"✅ 组件类已注册: {name} ({component_class.__name__})")
            return True

        except Exception as e:
            logger.error(f"❌ 组件类注册失败 {name}: {e}")
            return False

    def unregister_component(self, name: str) -> bool:
        """
        注销组件

        Args:
            name: 组件名称

        Returns:
            bool: 注销是否成功
        """
        try:
            if name not in self._components:
                logger.warning(f"组件 {name} 不存在")
                return False

            component_info = self._components[name]

            # 清理实例
            if name in self._instances:
                instance = self._instances[name]
                if hasattr(instance, 'cleanup'):
                    try:
                        asyncio.create_task(instance.cleanup())
                    except Exception as e:
                        logger.warning(f"清理组件实例失败 {name}: {e}")
                del self._instances[name]

            del self._components[name]

            # 触发注销事件
            asyncio.create_task(self._emit_event("component_unregistered", component_info))

            logger.info(f"✅ 组件已注销: {name}")
            return True

        except Exception as e:
            logger.error(f"❌ 组件注销失败 {name}: {e}")
            return False

    def get_component_info(self, name: str) -> Optional[ComponentInfo]:
        """
        获取组件信息

        Args:
            name: 组件名称

        Returns:
            Optional[ComponentInfo]: 组件信息
        """
        return self._components.get(name)

    def list_components(
        self,
        category: Optional[str] = None,
        initialized_only: bool = False
    ) -> List[str]:
        """
        列出组件

        Args:
            category: 分类过滤
            initialized_only: 仅列出已初始化组件

        Returns:
            List[str]: 组件名称列表
        """
        components = []

        for name, info in self._components.items():
            if category and info.category != category:
                continue
            if initialized_only and not info.initialized:
                continue
            components.append(name)

        return sorted(components, key=lambda x: self._components[x].priority, reverse=True)

    async def discover_components(self, search_paths: Optional[List[str]] = None) -> int:
        """
        自动发现组件

        Args:
            search_paths: 搜索路径列表

        Returns:
            int: 发现的组件数量
        """
        paths = search_paths or self._discovery_paths
        discovered_count = 0

        for path in paths:
            try:
                path_obj = Path(path)
                if not path_obj.exists():
                    logger.warning(f"搜索路径不存在: {path}")
                    continue

                # 搜索Python文件
                for py_file in path_obj.rglob("*.py"):
                    if py_file.name.startswith("__"):
                        continue

                    discovered = await self._discover_components_in_file(py_file)
                    discovered_count += discovered

            except Exception as e:
                logger.error(f"搜索路径失败 {path}: {e}")

        logger.info(f"🔍 自动发现完成，发现 {discovered_count} 个组件")
        return discovered_count

    async def _discover_components_in_file(self, file_path: Path) -> int:
        """在文件中发现组件"""
        discovered_count = 0

        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 检查模块中的类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # 跳过导入的类
                if obj.__module__ != module.__name__:
                    continue

                # 检查是否为组件类
                if (issubclass(obj, IComponent) and
                    obj != IComponent and
                    not inspect.isabstract(obj)):

                    # 尝试自动注册
                    if hasattr(obj, '__registry_info__'):
                        # 使用类上的注册信息
                        registry_info = obj.__registry_info__
                        success = await self.register_component_class(
                            obj,
                            **registry_info
                        )
                    else:
                        # 自动推断信息
                        success = await self.register_component_class(obj)

                    if success:
                        discovered_count += 1

        except Exception as e:
            logger.error(f"发现组件失败 {file_path}: {e}")

        return discovered_count

    async def initialize_component(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化组件

        Args:
            name: 组件名称
            config: 组件配置

        Returns:
            bool: 初始化是否成功
        """
        if name not in self._components:
            logger.error(f"组件 {name} 未注册")
            return False

        component_info = self._components[name]

        # 检查是否已初始化（单例模式）
        if component_info.singleton and component_info.initialized:
            logger.debug(f"组件 {name} 已初始化（单例模式）")
            return True

        try:
            # 合并配置
            final_config = component_info.config.copy()
            if config:
                final_config.update(config)

            # 检查依赖
            for dep_name in component_info.dependencies:
                if dep_name not in self._instances:
                    logger.warning(f"依赖组件 {dep_name} 未初始化，尝试初始化")
                    if not await self.initialize_component(dep_name):
                        raise RuntimeError(f"依赖组件 {dep_name} 初始化失败")

            # 创建实例
            instance = component_info.component_class()

            # 配置实例
            if hasattr(instance, 'update_configuration') and final_config:
                instance.update_configuration(final_config)

            # 初始化实例
            success = await instance.initialize(final_config)
            if not success:
                raise RuntimeError("组件初始化失败")

            # 存储实例
            if component_info.singleton:
                self._instances[name] = instance

            component_info.instance = instance
            component_info.initialized = True

            # 触发初始化事件
            await self._emit_event("component_initialized", component_info)

            logger.info(f"✅ 组件初始化成功: {name}")
            return True

        except Exception as e:
            logger.error(f"❌ 组件初始化失败 {name}: {e}")
            await self._emit_event("component_failed", component_info, error=e)
            return False

    async def get_component(self, name: str) -> Optional[IComponent]:
        """
        获取组件实例

        Args:
            name: 组件名称

        Returns:
            Optional[IComponent]: 组件实例
        """
        # 检查是否已有实例
        if name in self._instances:
            return self._instances[name]

        # 检查是否需要初始化
        component_info = self._components.get(name)
        if component_info and not component_info.initialized:
            await self.initialize_component(name)
            return self._instances.get(name)

        return None

    async def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        初始化所有组件

        Args:
            config: 全局配置

        Returns:
            Dict[str, bool]: 初始化结果
        """
        if self._state == RegistryState.INITIALIZING:
            logger.warning("注册器正在初始化中")
            return {}

        self._state = RegistryState.INITIALIZING
        results = {}

        try:
            # 计算初始化顺序（依赖关系拓扑排序）
            init_order = self._calculate_initialization_order()

            # 按顺序初始化
            for component_name in init_order:
                results[component_name] = await self.initialize_component(
                    component_name,
                    config.get(component_name) if config else None
                )

            self._state = RegistryState.READY
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"🚀 组件初始化完成: {success_count}/{len(results)} 成功")

        except Exception as e:
            logger.error(f"❌ 组件批量初始化失败: {e}")
            self._state = RegistryState.ERROR

        return results

    def _calculate_initialization_order(self) -> List[str]:
        """计算初始化顺序（拓扑排序）"""
        # 简单实现：按优先级排序
        components = list(self._components.items())
        components.sort(key=lambda x: x[1].priority, reverse=True)
        return [name for name, _ in components]

    def add_event_handler(self, event: str, handler: Callable) -> None:
        """添加事件处理器"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _emit_event(self, event: str, component_info: ComponentInfo, error: Optional[Exception] = None) -> None:
        """触发事件"""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(component_info, error)
                else:
                    handler(component_info, error)
            except Exception as e:
                logger.error(f"事件处理器失败 {event}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取注册器统计信息"""
        category_stats = {}
        for info in self._components.values():
            if info.category not in category_stats:
                category_stats[info.category] = {"total": 0, "initialized": 0}
            category_stats[info.category]["total"] += 1
            if info.initialized:
                category_stats[info.category]["initialized"] += 1

        return {
            "state": self._state.value,
            "total_components": len(self._components),
            "initialized_components": len(self._instances),
            "categories": category_stats,
            "discovery_paths": self._discovery_paths
        }

    async def cleanup_all(self) -> None:
        """清理所有组件"""
        for name in list(self._instances.keys()):
            instance = self._instances[name]
            if hasattr(instance, 'cleanup'):
                try:
                    await instance.cleanup()
                except Exception as e:
                    logger.warning(f"清理组件失败 {name}: {e}")

        self._instances.clear()
        for info in self._components.values():
            info.initialized = False
            info.instance = None

        self._state = RegistryState.UNINITIALIZED
        logger.info("🧹 所有组件已清理")


# 全局注册器实例
_global_registry: Optional[ComponentRegistry] = None


def get_global_registry() -> ComponentRegistry:
    """获取全局注册器实例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry


# 装饰器：自动注册组件
def register_component(**registry_kwargs):
    """组件注册装饰器"""
    def decorator(cls):
        # 延迟注册到全局注册器
        cls.__registry_info__ = registry_kwargs
        return cls
    return decorator


# 导出
__all__ = [
    'ComponentInfo',
    'ComponentCategory',
    'RegistryState',
    'ComponentRegistry',
    'get_global_registry',
    'register_component'
]