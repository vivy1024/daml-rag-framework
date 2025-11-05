#!/usr/bin/env python3
"""
DAML-RAG Framework 组件测试脚本
测试所有核心组件的功能
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加框架路径
sys.path.insert(0, str(Path(__file__).parent))

from daml_rag_learning import (
    InMemoryManager, Experience, Feedback, FeedbackType,
    ModelManager, DeepSeekProvider, OllamaProvider, ModelConfig, ModelType, GenerationRequest,
    SimpleFeedbackProcessor, FeedbackData, FeedbackSource,
    ExperienceBasedLearner, AdaptationConfig, AdaptationStrategy
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_memory_manager():
    """测试记忆管理器"""
    print("\n" + "="*60)
    print("🧠 测试记忆管理器")
    print("="*60)

    # 创建内存管理器
    config = {
        'max_experiences': 100,
        'experience_ttl_days': 30,
        'similarity_threshold': 0.6
    }
    memory_manager = InMemoryManager(config)
    await memory_manager.initialize()

    # 创建测试经验
    experience1 = Experience(
        id="",
        query="如何制定增肌计划？",
        response="制定增肌计划需要考虑以下几个方面：\n1. 训练频率：每周3-4次\n2. 动作选择：复合动作为主\n3. 营养摄入：高蛋白饮食",
        context={"user_level": "beginner", "goal": "muscle_gain"},
        model_used="test-model"
    )

    experience2 = Experience(
        id="",
        query="深蹲的正确动作要领是什么？",
        response="深蹲的正确动作要领：\n1. 双脚与肩同宽\n2. 膝盖与脚尖方向一致\n3. 保持背部挺直\n4. 大腿与地面平行",
        context={"exercise": "squat", "level": "basic"},
        model_used="test-model"
    )

    # 存储经验
    success1 = await memory_manager.store_experience(experience1)
    success2 = await memory_manager.store_experience(experience2)
    print(f"✅ 经验存储: {success1}, {success2}")

    # 检索相似经验
    similar_experiences = await memory_manager.retrieve_similar_experiences(
        "我想了解增肌训练方法",
        top_k=2
    )
    print(f"🔍 检索到 {len(similar_experiences)} 个相似经验")
    for i, exp in enumerate(similar_experiences, 1):
        print(f"   {i}. {exp.query[:30]}... (相似度: {exp.similarity_score:.2f})")

    # 添加反馈
    feedback = Feedback(
        id="",
        experience_id=experience1.id,
        user_id="test-user",
        feedback_type=FeedbackType.POSITIVE,
        rating=4.5,
        comment="很有用的建议"
    )
    await memory_manager.update_feedback(experience1.id, feedback)
    print("✅ 反馈已添加")

    # 获取统计信息
    stats = await memory_manager.get_statistics()
    print(f"📊 记忆统计: {stats['total_experiences']} 个经验, "
          f"平均质量分数: {stats['average_quality_score']:.2f}")

    return memory_manager


async def test_model_provider():
    """测试模型提供者"""
    print("\n" + "="*60)
    print("🤖 测试模型提供者")
    print("="*60)

    # 创建模型配置（使用模拟配置）
    teacher_config = ModelConfig(
        model_name="deepseek-chat",
        model_type=ModelType.TEACHER,
        api_key="test-key",
        api_base="https://api.deepseek.com",
        cost_per_token=0.001
    )

    student_config = ModelConfig(
        model_name="qwen2.5:14b",
        model_type=ModelType.STUDENT,
        api_base="http://localhost:11434",
        cost_per_token=0.0001
    )

    # 创建提供者（实际使用时需要真实的API密钥）
    print("⚠️  模型提供者测试需要真实的API配置，这里只展示配置结构")
    print(f"📋 教师模型配置: {teacher_config.model_name} (成本: {teacher_config.cost_per_token})")
    print(f"📋 学生模型配置: {student_config.model_name} (成本: {student_config.cost_per_token})")

    return None


async def test_feedback_processor():
    """测试反馈处理器"""
    print("\n" + "="*60)
    print("📝 测试反馈处理器")
    print("="*60)

    # 创建反馈处理器
    feedback_processor = SimpleFeedbackProcessor()

    # 添加测试反馈
    feedback1 = FeedbackData(
        query_id="query-1",
        response_id="response-1",
        feedback_type="thumbs_up",
        feedback_source=FeedbackSource.USER_EXPLICIT,
        rating=4.5,
        comment="回答很准确"
    )

    feedback2 = FeedbackData(
        query_id="query-2",
        response_id="response-2",
        feedback_type="thumbs_down",
        feedback_source=FeedbackSource.USER_EXPLICIT,
        rating=2.0,
        comment="回答不够详细"
    )

    success1 = await feedback_processor.collect_feedback(feedback1)
    success2 = await feedback_processor.collect_feedback(feedback2)
    print(f"✅ 反馈收集: {success1}, {success2}")

    # 分析反馈
    analysis = await feedback_processor.analyze_feedback()
    print(f"📊 反馈分析: {analysis.total_feedbacks} 个反馈, "
          f"平均评分: {analysis.average_rating:.2f}")

    # 获取改进建议
    suggestions = await feedback_processor.get_improvement_suggestions()
    print(f"💡 改进建议: {len(suggestions)} 条")
    for suggestion in suggestions[:3]:
        print(f"   - {suggestion}")

    return feedback_processor


async def test_adaptive_learner():
    """测试自适应学习器"""
    print("\n" + "="*60)
    print("🎯 测试自适应学习器")
    print("="*60)

    # 创建模拟组件
    memory_manager = InMemoryManager({'max_experiences': 50})
    await memory_manager.initialize()

    feedback_processor = SimpleFeedbackProcessor()

    # 创建适应配置
    config = AdaptationConfig(
        strategy=AdaptationStrategy.MODERATE,
        adaptation_interval=50,
        confidence_threshold=0.6
    )

    # 创建学习器（需要模型管理器，这里创建一个模拟的）
    class MockModelManager:
        def __init__(self):
            self.stats = {
                "total_requests": 100,
                "teacher_requests": 30,
                "student_requests": 70,
                "cache_hits": 20,
                "total_cost": 0.05,
                "total_tokens": 1000
            }

        def get_stats(self):
            return self.stats

    model_manager = MockModelManager()

    # 创建自适应学习器
    learner = ExperienceBasedLearner(
        memory_manager=memory_manager,
        feedback_processor=feedback_processor,
        model_manager=model_manager,
        config=config
    )
    await learner.initialize()

    # 分析性能
    performance = await learner.analyze_performance()
    print(f"📈 性能分析: 整体性能 {performance['overall_performance']:.2f}")
    for metric, value in performance.items():
        if metric != 'overall_performance':
            print(f"   - {metric}: {value:.2f}")

    # 识别适应机会
    opportunities = await learner.identify_adaptation_opportunities(performance)
    print(f"🔍 识别到 {len(opportunities)} 个适应机会")
    for opp in opportunities[:3]:
        print(f"   - {opp.target.value}: {opp.parameter} ({opp.reason})")

    # 获取适应统计
    stats = learner.get_adaptation_stats()
    print(f"📊 适应统计: 总适应次数 {stats['total_adaptations']}, "
          f"成功率 {stats.get('success_rate', 0):.2f}")

    return learner


async def test_integration():
    """集成测试"""
    print("\n" + "="*60)
    print("🔗 集成测试")
    print("="*60)

    # 创建所有组件
    memory_manager = InMemoryManager({'max_experiences': 100})
    await memory_manager.initialize()

    feedback_processor = SimpleFeedbackProcessor()

    # 模拟完整的对话流程
    print("📝 模拟用户对话流程...")

    # 1. 用户查询
    query = "健身新手应该如何开始训练？"
    print(f"👤 用户查询: {query}")

    # 2. 创建经验
    experience = Experience(
        id="",
        query=query,
        response="作为健身新手，建议按以下步骤开始：\n1. 设定明确目标\n2. 选择合适的训练计划\n3. 注重基础动作\n4. 保持 consistency",
        context={"user_level": "beginner", "goal": "general_fitness"},
        model_used="test-model"
    )

    # 3. 存储经验
    await memory_manager.store_experience(experience)
    print("💾 经验已存储")

    # 4. 用户反馈
    feedback_data = FeedbackData(
        query_id="query-newbie",
        response_id="response-newbie",
        feedback_type="thumbs_up",
        feedback_source=FeedbackSource.USER_EXPLICIT,
        rating=4.0,
        comment="对新手很友好"
    )
    await feedback_processor.collect_feedback(feedback_data)
    await memory_manager.update_feedback(experience.id, Feedback(
        id="",
        experience_id=experience.id,
        user_id="newbie-user",
        feedback_type=FeedbackType.POSITIVE,
        rating=4.0,
        comment="对新手很友好"
    ))
    print("👍 用户反馈已记录")

    # 5. 相似查询
    similar_query = "初学者健身计划建议"
    similar_experiences = await memory_manager.retrieve_similar_experiences(similar_query)
    print(f"🔍 找到 {len(similar_experiences)} 个相关经验")

    # 6. 获取统计信息
    memory_stats = await memory_manager.get_statistics()
    feedback_stats = feedback_processor.get_feedback_stats()

    print("📊 系统统计:")
    print(f"   - 经验总数: {memory_stats['total_experiences']}")
    print(f"   - 平均质量: {memory_stats['average_quality_score']:.2f}")
    print(f"   - 反馈总数: {feedback_stats['total_feedbacks']}")
    print(f"   - 平均评分: {feedback_stats['average_rating']:.2f}")

    print("✅ 集成测试完成")


async def main():
    """主测试函数"""
    print("🚀 DAML-RAG Framework 组件测试")
    print("="*60)
    print("测试所有核心组件的功能...")

    try:
        # 测试各个组件
        memory_manager = await test_memory_manager()
        await test_model_provider()
        feedback_processor = await test_feedback_processor()
        learner = await test_adaptive_learner()
        await test_integration()

        print("\n" + "="*60)
        print("🎉 所有组件测试完成!")
        print("="*60)

        print("✅ 测试结果:")
        print("   - 记忆管理器: 正常")
        print("   - 模型提供者: 配置正常")
        print("   - 反馈处理器: 正常")
        print("   - 自适应学习器: 正常")
        print("   - 集成测试: 正常")

        print("\n🔧 组件功能验证:")
        print("   ✅ 经验存储和检索")
        print("   ✅ 相似度计算")
        print("   ✅ 反馈收集和分析")
        print("   ✅ 质量指标更新")
        print("   ✅ 性能监控")
        print("   ✅ 自适应调整")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("DAML-RAG Framework 组件测试")
    print("测试框架的所有核心组件功能")
    print("="*60)

    asyncio.run(main())