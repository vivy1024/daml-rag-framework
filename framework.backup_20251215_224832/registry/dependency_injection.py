# -*- coding: utf-8 -*-
"""
DAML-RAG框架依赖注入系统 v2.0

提供灵活的依赖注入和IoC容器功能。

版本：v2.0.0
更新日期：2025-11-17
设计原则：松耦合、可配置、自动注入
"""

import inspect
import logging
from typing import Dict, Any, List, Optional, Type, Callable, Union, get_type_hints
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC

from ..interfaces.base import IComponent
from ..interfaces.retrieval import IRetriever
from ..interfaces.orchestration import IOrchestrator
from ..interfaces.quality import IQualityChecker
from ..interfaces.storage import IStorage

logger = logging.getLogger(__name__)


@dataclass
class DependencyDescriptor:
    """依赖描述符"""
    name: str
    interface_type: Optional[Type] = None
    required: bool = True
    default_value: Any = None
    factory: Optional[Callable] = None
    singleton: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class InjectionScope(Enum):
    """注入范围"""
    SINGLETON = "singleton"       # 单例
    TRANSIENT = "transient"       # 瞬时
    SCOPED = "scoped"           # 作用域


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    interface_type: Type
    implementation_type: Type
    scope: InjectionScope = InjectionScope.SINGLETON
    dependencies: List[DependencyDescriptor] = field(default_factory=list)
    factory: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    lazy: bool = False


class IContainer(ABC):
    """IoC容器接口"""

    @abstractmethod
    def register_singleton(self, interface: Type, implementation: Type, config: Optional[Dict[str, Any]] = None) -> None:
        """注册单例服务"""
        pass

    @abstractmethod
    def register_transient(self, interface: Type, implementation: Type, config: Optional[Dict[str, Any]] = None) -> None:
        """注册瞬态服务"""
        pass

    @abstractmethod
    def register_factory(self, interface: Type, factory: Callable, scope: InjectionScope = InjectionScope.SINGLETON) -> None:
        """注册工厂服务"""
        pass

    @abstractmethod
    def resolve(self, interface: Type) -> Any:
        """解析服务"""
        pass

    @abstractmethod
    def resolve_all(self) -> Dict[str, Any]:
        """解析所有服务"""
        pass

    @abstractmethod
    def is_registered(self, interface: Type) -> bool:
        """检查是否已注册"""
        pass

    @abstractmethod
    def create_scope(self) -> "IScope":
        """创建作用域"""
        pass


class IScope(ABC):
    """作用域接口"""

    @abstractmethod
    def resolve(self, interface: Type) -> Any:
        """在作用域中解析服务"""
        pass

    @abstractmethod
    def dispose(self) -> None:
        """释放作用域"""
        pass


class DIContainer(IContainer):
    """
    依赖注入容器

    实现IoC容器的核心功能。
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._current_scope: Optional[IScope] = None
        self._resolving_stack: List[Type] = field(default_factory=list)
        self._config: Dict[str, Any] = {}

    def set_config(self, config: Dict[str, Any]) -> None:
        """设置容器配置"""
        self._config.update(config)

    def register_singleton(self, interface: Type, implementation: Type, config: Optional[Dict[str, Any]] = None) -> None:
        """注册单例服务"""
        self._register_service(interface, implementation, InjectionScope.SINGLETON, config)

    def register_transient(self, interface: Type, implementation: Type, config: Optional[Dict[str, Any]] = None) -> None:
        """注册瞬态服务"""
        self._register_service(interface, implementation, InjectionScope.TRANSIENT, config)

    def register_factory(self, interface: Type, factory: Callable, scope: InjectionScope = InjectionScope.SINGLETON) -> None:
        """注册工厂服务"""
        descriptor = ServiceDescriptor(
            interface_type=interface,
            implementation_type=type(None),  # 工厂没有具体类型
            scope=scope,
            factory=factory
        )
        self._services[interface] = descriptor

    def _register_service(self, interface: Type, implementation: Type, scope: InjectionScope, config: Optional[Dict[str, Any]] = None) -> None:
        """注册服务"""
        if not issubclass(implementation, interface):
            raise ValueError(f"{implementation} 必须实现 {interface}")

        # 分析依赖
        dependencies = self._analyze_dependencies(implementation)

        descriptor = ServiceDescriptor(
            interface_type=interface,
            implementation_type=implementation,
            scope=scope,
            dependencies=dependencies,
            config=config or {}
        )
        self._services[interface] = descriptor

    def _analyze_dependencies(self, implementation_type: Type) -> List[DependencyDescriptor]:
        """分析类的依赖"""
        dependencies = []

        # 检查构造函数参数
        init_signature = inspect.signature(implementation_type.__init__)
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue

            dependency = self._create_dependency_descriptor(param_name, param.annotation, param.default)
            if dependency:
                dependencies.append(dependency)

        # 检查属性注入
        if hasattr(implementation_type, '__inject__'):
            inject_attrs = getattr(implementation_type, '__inject__', [])
            for attr_info in inject_attrs:
                if isinstance(attr_info, tuple):
                    name, annotation = attr_info
                else:
                    name = attr_info
                    annotation = None
                dependency = self._create_dependency_descriptor(name, annotation)
                if dependency:
                    dependencies.append(dependency)

        return dependencies

    def _create_dependency_descriptor(self, name: str, annotation: Any, default: Any = None) -> Optional[DependencyDescriptor]:
        """创建依赖描述符"""
        # 处理类型注解
        interface_type = None
        if annotation != inspect.Parameter.empty and annotation != inspect._empty:
            interface_type = annotation

        # 检查是否为必需依赖
        required = default == inspect.Parameter.empty
        default_value = None if required else default

        return DependencyDescriptor(
            name=name,
            interface_type=interface_type,
            required=required,
            default_value=default_value
        )

    def resolve(self, interface: Type) -> Any:
        """解析服务"""
        if interface not in self._services:
            raise ValueError(f"服务 {interface} 未注册")

        # 检查循环依赖
        if interface in self._resolving_stack:
            cycle = " -> ".join([cls.__name__ for cls in self._resolving_stack] + [interface.__name__])
            raise RuntimeError(f"检测到循环依赖: {cycle}")

        descriptor = self._services[interface]

        # 检查作用域缓存
        if self._current_scope and descriptor.scope == InjectionScope.SCOPED:
            cached = self._current_scope.get_cached(interface)
            if cached is not None:
                return cached

        # 单例缓存
        if descriptor.scope == InjectionScope.SINGLETON and interface in self._singletons:
            return self._singletons[interface]

        try:
            self._resolving_stack.append(interface)

            # 创建实例
            if descriptor.factory:
                instance = descriptor.factory()
            else:
                instance = self._create_instance(descriptor)

            # 配置实例
            if hasattr(instance, 'configure') and descriptor.config:
                instance.configure(descriptor.config)

            # 属性注入
            self._inject_properties(instance, descriptor)

            # 缓存实例
            if descriptor.scope == InjectionScope.SINGLETON:
                self._singletons[interface] = instance
            elif self._current_scope and descriptor.scope == InjectionScope.SCOPED:
                self._current_scope.cache(interface, instance)

            return instance

        finally:
            self._resolving_stack.pop()

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建实例"""
        # 解析依赖
        dependency_instances = {}
        for dep in descriptor.dependencies:
            if dep.interface_type:
                dependency_instances[dep.name] = self.resolve(dep.interface_type)
            elif dep.default_value is not None:
                dependency_instances[dep.name] = dep.default_value
            elif dep.required:
                raise ValueError(f"必需依赖 {dep.name} 未提供")

        # 创建实例
        implementation_type = descriptor.implementation_type
        return implementation_type(**dependency_instances)

    def _inject_properties(self, instance: Any, descriptor: ServiceDescriptor) -> None:
        """属性注入"""
        if hasattr(instance, '__inject__'):
            inject_attrs = getattr(instance, '__inject__', [])
            for attr_info in inject_attrs:
                if isinstance(attr_info, tuple):
                    name, annotation = attr_info
                    if annotation and annotation != inspect._empty:
                        resolved = self.resolve(annotation)
                        setattr(instance, name, resolved)
                else:
                    name = attr_info
                    # 尝试从类型提示解析
                    if hasattr(instance, '__annotations__'):
                        annotations = getattr(instance, '__annotations__')
                        if name in annotations:
                            resolved = self.resolve(annotations[name])
                            setattr(instance, name, resolved)

    def resolve_all(self) -> Dict[str, Any]:
        """解析所有服务"""
        resolved = {}
        for interface in self._services:
            try:
                instance = self.resolve(interface)
                resolved[interface.__name__] = instance
            except Exception as e:
                logger.error(f"解析服务失败 {interface.__name__}: {e}")
        return resolved

    def is_registered(self, interface: Type) -> bool:
        """检查是否已注册"""
        return interface in self._services

    def create_scope(self) -> "IScope":
        """创建作用域"""
        return Scope(self)


class Scope(IScope):
    """作用域实现"""

    def __init__(self, container: DIContainer):
        self._container = container
        self._container._current_scope = self
        self._scoped_instances: Dict[Type, Any] = {}
        self._disposed = False

    def resolve(self, interface: Type) -> Any:
        """在作用域中解析服务"""
        if self._disposed:
            raise RuntimeError("作用域已释放")

        # 检查作用域缓存
        if interface in self._scoped_instances:
            return self._scoped_instances[interface]

        # 从容器解析
        instance = self._container.resolve(interface)
        self._scoped_instances[interface] = instance
        return instance

    def get_cached(self, interface: Type) -> Any:
        """获取缓存的实例"""
        return self._scoped_instances.get(interface)

    def cache(self, interface: Type, instance: Any) -> None:
        """缓存实例"""
        if not self._disposed:
            self._scoped_instances[interface] = instance

    def dispose(self) -> None:
        """释放作用域"""
        if not self._disposed:
            # 清理瞬态实例
            for interface, instance in self._scoped_instances.items():
                if hasattr(instance, 'cleanup'):
                    try:
                        import asyncio
                        if asyncio.iscoroutinefunction(instance.cleanup):
                            asyncio.create_task(instance.cleanup())
                        else:
                            instance.cleanup()
                    except Exception as e:
                        logger.warning(f"清理实例失败 {interface}: {e}")

            self._scoped_instances.clear()
            self._container._current_scope = None
            self._disposed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()


# 装饰器：依赖注入
def inject(container: Optional[DIContainer] = None):
    """依赖注入装饰器"""
    def decorator(cls):
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            if container is None:
                # 使用全局容器
                global _global_container
                if '_global_container' not in globals():
                    _global_container = DIContainer()
                container = _global_container

            # 自动注入依赖
            if not args:
                resolved_args = []
                init_signature = inspect.signature(original_init)
                for param_name, param in init_signature.parameters.items():
                    if param_name == 'self':
                        continue
                    if param.annotation != inspect.Parameter.empty:
                        try:
                            resolved = container.resolve(param.annotation)
                            resolved_args.append(resolved)
                        except Exception:
                            if param.default == inspect.Parameter.empty:
                                raise
                            resolved_args.append(param.default)

                original_init(self, *resolved_args, **kwargs)
            else:
                original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return decorator


# 装饰器：自动注册
def auto_register(container: Optional[DIContainer] = None, scope: InjectionScope = InjectionScope.SINGLETON):
    """自动注册装饰器"""
    def decorator(cls):
        if container is None:
            global _global_container
            if '_global_container' not in globals():
                _global_container = DIContainer()
            container = _global_container

        # 自动注册为服务
        if hasattr(cls, '__interface__'):
            interface = getattr(cls, '__interface__')
            container.register_singleton(interface, cls)
        else:
            # 使用类本身作为接口
            container.register_singleton(cls, cls)

        # 应用依赖注入
        return inject(container)(cls)

    return decorator


# 全局容器实例
_global_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """获取全局容器实例"""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


# 导出
__all__ = [
    'DependencyDescriptor',
    'InjectionScope',
    'ServiceDescriptor',
    'IContainer',
    'IScope',
    'DIContainer',
    'Scope',
    'inject',
    'auto_register',
    'get_container'
]