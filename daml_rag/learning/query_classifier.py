"""
QueryComplexityClassifier - 查询复杂度分类器（通用框架）

基于BGE向量模型的语义相似度分类

理论基础：
1. Semantic Similarity: 使用向量相似度判断查询复杂度
2. BGE (BAAI General Embedding): 中文语义向量模型
3. Cosine Similarity: 余弦相似度度量语义距离
4. Fallback Strategy: 硬编码关键词作为兜底策略

设计原则：
- 领域无关：不依赖特定业务逻辑
- 可扩展：支持自定义复杂查询向量库
- 高效：向量计算缓存，避免重复编码

数学原理：
    余弦相似度：
    similarity = (A · B) / (||A|| * ||B||)
    
    分类规则：
    - similarity > 0.7  → 复杂查询 → Teacher Model
    - similarity < 0.5  → 简单查询 → Student Model
    - 0.5 ≤ similarity ≤ 0.7 → 中等查询 → Context-Dependent

Example:
    >>> from daml_rag.learning import QueryComplexityClassifier
    >>> 
    >>> classifier = QueryComplexityClassifier()
    >>> 
    >>> # 分类查询
    >>> is_complex, similarity, reason = classifier.classify_complexity(
    ...     query="帮我设计一套增肌训练计划，我有腰椎间盘突出"
    ... )
    >>> print(f"复杂度: {is_complex}, 相似度: {similarity:.2f}, 理由: {reason}")
    >>> # 输出: 复杂度: True, 相似度: 0.85, 理由: 与复杂查询示例高度相似

版本：v1.1.0
日期：2025-11-07
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class QueryComplexityClassifier:
    """
    查询复杂度分类器
    
    核心算法：BGE向量模型 + 余弦相似度
    
    工作流程：
    1. 加载BGE模型（首次调用）
    2. 编码查询为向量
    3. 计算与复杂查询库的相似度
    4. 根据阈值分类
    5. 返回结果和理由
    
    设计原则：
    - 懒加载：首次使用时才加载模型
    - 缓存：复杂查询向量库预计算
    - 降级：模型加载失败时使用硬编码关键词
    
    Example:
        >>> classifier = QueryComplexityClassifier(
        ...     complex_query_examples=[
        ...         "帮我设计一套增肌训练计划，我有腰椎间盘突出",
        ...         "三个月减脂10公斤，需要详细的营养和训练方案"
        ...     ],
        ...     similarity_threshold=0.7
        ... )
        >>> 
        >>> is_complex, sim, reason = classifier.classify_complexity(
        ...     "设计训练计划"
        ... )
    """
    
    def __init__(
        self,
        complex_query_examples: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
        moderate_threshold: float = 0.5,
        use_fallback_keywords: bool = True,
        model_name: str = "BAAI/bge-base-zh-v1.5"
    ):
        """
        初始化分类器
        
        Args:
            complex_query_examples: 复杂查询示例列表（用于构建向量库）
            similarity_threshold: 高相似度阈值（>= 则判定为复杂）
            moderate_threshold: 低相似度阈值（< 则判定为简单）
            use_fallback_keywords: 是否使用硬编码关键词作为兜底
            model_name: BGE模型名称（默认 BAAI/bge-base-zh-v1.5）
        """
        self.similarity_threshold = similarity_threshold
        self.moderate_threshold = moderate_threshold
        self.use_fallback_keywords = use_fallback_keywords
        self.model_name = model_name
        
        # 模型懒加载（首次调用时加载）
        self._model = None
        self._model_load_failed = False
        
        # 复杂查询向量库（懒计算）
        self.complex_query_examples = complex_query_examples or self._get_default_examples()
        self._complex_query_vectors = None
        
        # 硬编码关键词（兜底策略）
        self.complex_keywords = [
            '计划', '训练', '方案', '设计', '康复', '个性化',
            '增肌', '减脂', '力量', '周期', '损伤', '营养',
            'plan', 'training', 'program', 'design', 'personalized'
        ]
        
        logger.info(
            f"QueryComplexityClassifier initialized: "
            f"model={model_name}, "
            f"threshold={similarity_threshold}, "
            f"examples={len(self.complex_query_examples)}"
        )
    
    def _get_default_examples(self) -> List[str]:
        """
        获取默认的复杂查询示例库
        
        Returns:
            List[str]: 复杂查询示例列表
        """
        return [
            # 训练计划类（高复杂度）
            "帮我设计一套增肌训练计划，我有腰椎间盘突出，需要避免压迫脊柱的动作",
            "三个月减脂10公斤，需要详细的营养和训练方案，包括每周训练安排和饮食计划",
            "制定一个适合新手的全身力量训练计划，一周训练3天，每次60分钟",
            
            # 康复方案类（高复杂度）
            "肩关节损伤后如何恢复训练？需要具体的康复动作和训练强度建议",
            "膝盖受伤康复期间，如何继续进行力量训练而不影响恢复？",
            "腰椎间盘突出患者的康复训练方案，包括禁忌动作和安全替代方案",
            
            # 营养设计类（高复杂度）
            "设计一份增肌期的详细营养计划，包括每日热量分配和食物选择",
            "如何制定减脂期的饮食方案？需要考虑蛋白质摄入和热量赤字",
            "素食人群的增肌营养方案，如何保证蛋白质摄入充足？",
            
            # 综合咨询类（高复杂度）
            "全面的健身指导，包括训练、营养、补剂、休息等各方面建议",
            "从零开始健身，需要完整的指导方案，包括目标设定、计划制定、动作学习",
            
            # 周期化训练类（高复杂度）
            "如何进行周期化训练？需要详细的分期计划和强度安排",
            "力量周期训练的设计原则和实施方案",
            
            # 特殊人群类（高复杂度）
            "中老年人的力量训练方案，需要注意哪些事项？",
            "孕期和产后的安全训练计划，如何恢复核心力量？"
        ]
    
    def _load_model(self):
        """
        懒加载BGE模型
        
        如果模型加载失败，标记失败状态，后续使用硬编码关键词兜底
        """
        if self._model is not None:
            return  # 已加载
        
        if self._model_load_failed:
            return  # 之前加载失败过，不再尝试
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"加载BGE模型: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("✅ BGE模型加载成功")
            
            # 预计算复杂查询向量库
            self._precompute_complex_vectors()
            
        except Exception as e:
            logger.warning(
                f"⚠️ BGE模型加载失败: {e}，将使用硬编码关键词兜底策略"
            )
            self._model_load_failed = True
            self._model = None
    
    def _precompute_complex_vectors(self):
        """
        预计算复杂查询向量库
        
        将所有复杂查询示例编码为向量，缓存起来，避免重复计算
        """
        if self._model is None or self._complex_query_vectors is not None:
            return
        
        try:
            logger.info(
                f"预计算复杂查询向量库（{len(self.complex_query_examples)} 条）..."
            )
            self._complex_query_vectors = self._model.encode(
                self.complex_query_examples,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2归一化，便于余弦相似度计算
            )
            logger.info("✅ 复杂查询向量库预计算完成")
            
        except Exception as e:
            logger.error(f"❌ 复杂查询向量库预计算失败: {e}")
            self._complex_query_vectors = None
    
    def classify_complexity(
        self,
        query: str
    ) -> Tuple[bool, float, str]:
        """
        分类查询复杂度
        
        Args:
            query: 用户查询
        
        Returns:
            Tuple[bool, float, str]:
                - is_complex: 是否复杂查询
                - similarity: 最大相似度（如果使用向量模型）
                - reason: 分类理由
        
        Example:
            >>> is_complex, sim, reason = classifier.classify_complexity(
            ...     "设计增肌训练计划"
            ... )
            >>> print(f"复杂: {is_complex}, 相似度: {sim:.2f}")
        """
        # 1. 尝试使用BGE模型
        if not self._model_load_failed:
            self._load_model()
        
        # 2. 如果模型可用，使用向量相似度
        if self._model is not None and self._complex_query_vectors is not None:
            return self._classify_by_vector(query)
        
        # 3. 降级到硬编码关键词
        if self.use_fallback_keywords:
            return self._classify_by_keywords(query)
        
        # 4. 无法分类，默认为复杂（保守策略）
        logger.warning("⚠️ 无法分类查询复杂度，默认使用教师模型")
        return True, 0.0, "无法分类，默认使用教师模型（保守策略）"
    
    def _classify_by_vector(
        self,
        query: str
    ) -> Tuple[bool, float, str]:
        """
        基于向量相似度分类
        
        Args:
            query: 用户查询
        
        Returns:
            Tuple[bool, float, str]: (是否复杂, 最大相似度, 理由)
        """
        try:
            # 编码查询
            query_vector = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            # 计算与复杂查询库的相似度（已归一化，直接点积即为余弦相似度）
            similarities = np.dot(self._complex_query_vectors, query_vector)
            max_similarity = float(np.max(similarities))
            max_idx = int(np.argmax(similarities))
            
            # 根据阈值分类
            if max_similarity >= self.similarity_threshold:
                reason = (
                    f"与复杂查询示例高度相似（相似度={max_similarity:.2f}）: "
                    f"'{self.complex_query_examples[max_idx][:30]}...'"
                )
                logger.info(f"✅ [BGE分类] 复杂查询（相似度={max_similarity:.2f}）")
                return True, max_similarity, reason
            
            elif max_similarity < self.moderate_threshold:
                reason = f"与复杂查询示例相似度低（{max_similarity:.2f}）"
                logger.info(f"✅ [BGE分类] 简单查询（相似度={max_similarity:.2f}）")
                return False, max_similarity, reason
            
            else:
                # 中等相似度，需要其他因素判断
                reason = (
                    f"中等复杂度（相似度={max_similarity:.2f}），"
                    "建议结合Few-Shot判断"
                )
                logger.info(
                    f"⚠️ [BGE分类] 中等复杂度（相似度={max_similarity:.2f}），"
                    "需要额外判断"
                )
                # 中等复杂度时，偏向保守，使用教师模型
                return True, max_similarity, reason
        
        except Exception as e:
            logger.error(f"❌ BGE向量分类失败: {e}，降级到关键词匹配")
            return self._classify_by_keywords(query)
    
    def _classify_by_keywords(
        self,
        query: str
    ) -> Tuple[bool, float, str]:
        """
        基于硬编码关键词分类（兜底策略）
        
        Args:
            query: 用户查询
        
        Returns:
            Tuple[bool, float, str]: (是否复杂, 相似度=0.0, 理由)
        """
        query_lower = query.lower()
        
        for keyword in self.complex_keywords:
            if keyword in query_lower:
                reason = f"匹配复杂查询关键词: '{keyword}'"
                logger.info(f"✅ [关键词兜底] 复杂查询（关键词='{keyword}'）")
                return True, 0.0, reason
        
        reason = "未匹配任何复杂查询关键词"
        logger.info("✅ [关键词兜底] 简单查询")
        return False, 0.0, reason
    
    def add_complex_example(self, example: str):
        """
        添加复杂查询示例到向量库
        
        Args:
            example: 复杂查询示例
        
        Note:
            添加后需要重新预计算向量库
        """
        if example not in self.complex_query_examples:
            self.complex_query_examples.append(example)
            logger.info(f"新增复杂查询示例: {example[:50]}...")
            
            # 重新预计算向量库
            if self._model is not None:
                self._complex_query_vectors = None
                self._precompute_complex_vectors()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取分类器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "model_loaded": self._model is not None,
            "model_name": self.model_name if self._model else "N/A",
            "complex_examples_count": len(self.complex_query_examples),
            "similarity_threshold": self.similarity_threshold,
            "moderate_threshold": self.moderate_threshold,
            "use_fallback": self.use_fallback_keywords,
            "vector_cache_ready": self._complex_query_vectors is not None
        }

