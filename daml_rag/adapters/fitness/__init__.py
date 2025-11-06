"""
健身领域适配器
"""

from .fitness_adapter import FitnessDomainAdapter
from .tools import *
from .knowledge import *
from .intent_matcher import FitnessIntentMatcher

__all__ = [
    "FitnessDomainAdapter",
    "FitnessIntentMatcher",
]