# -*- coding: utf-8 -*-
"""
DAML-RAG框架注册系统 v2.0

提供组件注册、依赖注入、生命周期管理等核心功能。

版本：v2.0.0
更新日期：2025-11-17
"""

from .component_registry import (
    ComponentInfo,
    ComponentCategory,
    RegistryState,
    ComponentRegistry,
    get_global_registry,
    register_component
)

from .dependency_injection import (
    DependencyDescriptor,
    InjectionScope,
    ServiceDescriptor,
    IContainer,
    IScope,
    DIContainer,
    Scope,
    inject,
    auto_register,
    get_container
)

__all__ = [
    # 组件注册
    'ComponentInfo',
    'ComponentCategory',
    'RegistryState',
    'ComponentRegistry',
    'get_global_registry',
    'register_component',

    # 依赖注入
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

__version__ = "2.0.0"
__author__ = "DAML-RAG Team"