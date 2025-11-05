"""
基础工具类和组件注册表
"""

from typing import Dict, Any, Type, Optional, List
import logging
from abc import ABC


class ComponentRegistry:
    """组件注册表"""

    def __init__(self):
        self._components: Dict[Type, Any] = {}
        self._component_names: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def register_component(self, component_class: Type, component: Any, name: Optional[str] = None):
        """注册组件"""
        self._components[component_class] = component
        if name:
            self._component_names[name] = component
        self.logger.debug(f"注册组件: {component_class.__name__}")

    def get_component(self, component_class: Type) -> Optional[Any]:
        """获取组件"""
        return self._components.get(component_class)

    def get_component_by_name(self, name: str) -> Optional[Any]:
        """通过名称获取组件"""
        return self._component_names.get(name)

    def list_components(self) -> List[Type]:
        """列出所有注册的组件类型"""
        return list(self._components.keys())

    def clear(self):
        """清空注册表"""
        self._components.clear()
        self._component_names.clear()


class BaseComponent(ABC):
    """组件基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False

    async def initialize(self) -> None:
        """初始化组件"""
        if self._initialized:
            return

        try:
            await self._do_initialize()
            self._initialized = True
            self.logger.info(f"组件 {self.__class__.__name__} 初始化完成")
        except Exception as e:
            self.logger.error(f"组件 {self.__class__.__name__} 初始化失败: {str(e)}")
            raise

    async def _do_initialize(self) -> None:
        """执行具体的初始化逻辑，子类重写"""
        pass

    async def cleanup(self) -> None:
        """清理组件资源"""
        if not self._initialized:
            return

        try:
            await self._do_cleanup()
            self._initialized = False
            self.logger.info(f"组件 {self.__class__.__name__} 清理完成")
        except Exception as e:
            self.logger.error(f"组件 {self.__class__.__name__} 清理失败: {str(e)}")

    async def _do_cleanup(self) -> None:
        """执行具体的清理逻辑，子类重写"""
        pass

    async def health_check(self) -> bool:
        """健康检查"""
        return self._initialized

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized


class ConfigurableComponent(BaseComponent):
    """可配置组件基类"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置，子类可重写"""
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value


class MonitorableComponent(ConfigurableComponent):
    """可监控组件基类"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._metrics: Dict[str, Any] = {}
        self._enabled_metrics = config.get('enabled_metrics', [])

    def record_metric(self, name: str, value: Any) -> None:
        """记录指标"""
        if not self._enabled_metrics or name in self._enabled_metrics:
            self._metrics[name] = value

    def get_metric(self, name: str) -> Any:
        """获取指标"""
        return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        """清空指标"""
        self._metrics.clear()


class ComponentFactory:
    """组件工厂"""

    def __init__(self):
        self._registry: Dict[str, Type] = {}

    def register(self, name: str, component_class: Type) -> None:
        """注册组件类"""
        self._registry[name] = component_class

    def create(self, name: str, config: Dict[str, Any]) -> BaseComponent:
        """创建组件实例"""
        if name not in self._registry:
            raise ValueError(f"未注册的组件类型: {name}")

        component_class = self._registry[name]
        return component_class(config)

    def create_all(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseComponent]:
        """批量创建组件"""
        components = {}
        for name, config in configs.items():
            components[name] = self.create(name, config)
        return components