# -*- coding: utf-8 -*-
"""
DAML-RAG框架基础接口定义 v2.0

定义框架中最基础的组件接口，所有其他接口都基于这些基础接口扩展。

版本：v2.0.0
更新日期：2025-11-17
设计原则：最小化依赖、最大化可扩展性
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import asyncio


@dataclass
class ComponentStatus:
    """组件状态"""
    name: str
    initialized: bool = False
    healthy: bool = False
    last_check: Optional[str] = None
    metadata: Dict[str, Any] = None


class ComponentState(Enum):
    """组件状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class IComponent(ABC):
    """
    基础组件接口

    所有DAML-RAG框架组件都必须实现此接口。
    提供组件生命周期管理和基本功能。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """组件名称"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """组件版本"""
        pass

    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化组件

        Args:
            config: 组件配置

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """清理组件资源"""
        pass

    @abstractmethod
    async def health_check(self) -> ComponentStatus:
        """
        健康检查

        Returns:
            ComponentStatus: 组件健康状态
        """
        pass

    @abstractmethod
    def get_state(self) -> ComponentState:
        """
        获取组件状态

        Returns:
            ComponentState: 当前组件状态
        """
        pass

    async def restart(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        重启组件

        Args:
            config: 可选的新配置

        Returns:
            bool: 重启是否成功
        """
        try:
            await self.cleanup()
            return await self.initialize(config)
        except Exception:
            return False


class IConfigurable(ABC):
    """
    可配置组件接口

    提供配置管理和动态配置更新功能。
    """

    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """
        获取当前配置

        Returns:
            Dict[str, Any]: 当前配置
        """
        pass

    @abstractmethod
    def update_configuration(self, config: Dict[str, Any]) -> bool:
        """
        更新配置

        Args:
            config: 新配置

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        验证配置

        Args:
            config: 待验证的配置

        Returns:
            bool: 配置是否有效
        """
        pass

    def get_default_configuration(self) -> Dict[str, Any]:
        """
        获取默认配置

        Returns:
            Dict[str, Any]: 默认配置
        """
        return {}


class IMonitorable(ABC):
    """
    可监控组件接口

    提供性能监控、指标收集和统计功能。
    """

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取组件指标

        Returns:
            Dict[str, Any]: 组件性能指标
        """
        pass

    @abstractmethod
    async def collect_statistics(self) -> Dict[str, Any]:
        """
        收集统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    def get_health_summary(self) -> Dict[str, Any]:
        """
        获取健康摘要

        Returns:
            Dict[str, Any]: 健康摘要信息
        """
        return {
            "status": "unknown",
            "metrics_available": False,
            "last_check": None
        }


class ILifecycleAware(ABC):
    """
    生命周期感知接口

    提供组件生命周期事件处理能力。
    """

    @abstractmethod
    async def on_startup(self) -> None:
        """启动时回调"""
        pass

    @abstractmethod
    async def on_shutdown(self) -> None:
        """关闭时回调"""
        pass

    @abstractmethod
    async def on_error(self, error: Exception) -> None:
        """
        错误处理回调

        Args:
            error: 发生的错误
        """
        pass

    @abstractmethod
    async def on_configuration_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """
        配置变更回调

        Args:
            old_config: 旧配置
            new_config: 新配置
        """
        pass


class IAsyncComponent(IComponent):
    """
    异步组件接口

    扩展基础组件接口，支持异步操作。
    """

    @abstractmethod
    async def start(self) -> bool:
        """
        启动组件

        Returns:
            bool: 启动是否成功
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """
        停止组件

        Returns:
            bool: 停止是否成功
        """
        pass

    @abstractmethod
    async def pause(self) -> bool:
        """
        暂停组件

        Returns:
            bool: 暂停是否成功
        """
        pass

    @abstractmethod
    async def resume(self) -> bool:
        """
        恢复组件

        Returns:
            bool: 恢复是否成功
        """
        pass


# 便捷的基础组件类
class BaseComponent(IComponent, IConfigurable, IMonitorable, ILifecycleAware):
    """
    基础组件实现类

    提供接口的默认实现，子类只需实现特定的业务逻辑。
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._state = ComponentState.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self._initialization_time: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def get_state(self) -> ComponentState:
        return self._state

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """默认初始化实现"""
        try:
            self._state = ComponentState.INITIALIZING

            if config:
                if self.validate_configuration(config):
                    self._config.update(config)
                else:
                    raise ValueError("Invalid configuration")

            # 调用启动回调
            await self.on_startup()

            self._state = ComponentState.READY
            self._initialization_time = asyncio.get_event_loop().time()
            return True

        except Exception as e:
            self._state = ComponentState.ERROR
            await self.on_error(e)
            return False

    async def cleanup(self) -> None:
        """默认清理实现"""
        try:
            self._state = ComponentState.STOPPING
            await self.on_shutdown()
            self._state = ComponentState.STOPPED
        except Exception as e:
            await self.on_error(e)

    async def health_check(self) -> ComponentStatus:
        """默认健康检查实现"""
        return ComponentStatus(
            name=self._name,
            initialized=(self._state != ComponentState.UNINITIALIZED),
            healthy=(self._state == ComponentState.READY),
            last_check=str(asyncio.get_event_loop().time()),
            metadata={"state": self._state.value}
        )

    def get_configuration(self) -> Dict[str, Any]:
        return self._config.copy()

    def update_configuration(self, config: Dict[str, Any]) -> bool:
        if self.validate_configuration(config):
            old_config = self._config.copy()
            self._config.update(config)
            # 异步处理配置变更
            asyncio.create_task(self.on_configuration_change(old_config, self._config))
            return True
        return False

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        return isinstance(config, dict)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "state": self._state.value,
            "initialization_time": self._initialization_time,
            "config_keys": list(self._config.keys()),
            **self._metrics
        }

    async def collect_statistics(self) -> Dict[str, Any]:
        return self.get_metrics()

    # 默认的生命周期回调实现（空实现）
    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    async def on_error(self, error: Exception) -> None:
        pass

    async def on_configuration_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        pass


# 导出接口
__all__ = [
    'IComponent',
    'IConfigurable',
    'IMonitorable',
    'ILifecycleAware',
    'IAsyncComponent',
    'ComponentStatus',
    'ComponentState',
    'BaseComponent'
]