"""
领域适配器模块
"""
from .base.adapter import BaseDomainAdapter
from .fitness.fitness_adapter import FitnessDomainAdapter

__all__ = [
    "BaseDomainAdapter",
    "FitnessDomainAdapter",
]


