#!/usr/bin/env python3
"""
daml-rag-framework 简化测试脚本
测试核心组件功能
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 添加框架路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 直接导入模块
sys.path.insert(0, str(current_dir / "玉珍健身-learning"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_memory_component():
    """测试记忆组件"""
    print("\n" + "="*60)
    print("🧠 测试记忆组件")
    print("="*60)

    try:
        # 导入记忆模块
        from memory import InMemoryManager, Experience, Feedback, FeedbackType

        # 创建记忆管理器
        config = {
            'max_experiences': 100,
            'similarity_threshold': 0.6
        }
        memory_manager = InMemoryManager(config)
        await memory_manager.initialize()

        # 创建测试经验
        experience = Experience(
            id="",
            query="如何制定增肌计划？",
            response="制定增肌计划需要考虑训练频率、动作选择、营养摄入等...",
            context={"user_level": "beginner", "goal": "muscle_gain"},
            model_used="test-model"
        )

        # 存储经验
        success = await memory_manager.store_experience(experience)
        print(f"✅ 经验存储: {success}")

        # 检索相似经验
        similar_experiences = await memory_manager.retrieve_similar_experiences(
            "我想了解增肌方法",
            top_k=2
        )
        print(f"🔍 检索到 {len(similar_experiences)} 个相似经验")

        # 添加反馈
        feedback = Feedback(
            id="",
            experience_id=experience.id,
            user_id="test-user",
            feedback_type=FeedbackType.POSITIVE,
            rating=4.5,
            comment="很有用的建议"
        )
        await memory_manager.update_feedback(experience.id, feedback)
        print("✅ 反馈已添加")

        # 获取统计信息
        stats = await memory_manager.get_statistics()
        print(f"📊 记忆统计: {stats['total_experiences']} 个经验")

        return True

    except Exception as e:
        print(f"❌ 记忆组件测试失败: {e}")
        return False


async def test_feedback_component():
    """测试反馈组件"""
    print("\n" + "="*60)
    print("📝 测试反馈组件")
    print("="*60)

    try:
        # 导入反馈模块
        from feedback import SimpleFeedbackProcessor, FeedbackData, FeedbackSource

        # 创建反馈处理器
        feedback_processor = SimpleFeedbackProcessor()

        # 添加测试反馈
        feedback = FeedbackData(
            query_id="query-1",
            response_id="response-1",
            feedback_type="thumbs_up",
            feedback_source=FeedbackSource.USER_EXPLICIT,
            rating=4.5,
            comment="回答很准确"
        )

        success = await feedback_processor.collect_feedback(feedback)
        print(f"✅ 反馈收集: {success}")

        # 分析反馈
        analysis = await feedback_processor.analyze_feedback()
        print(f"📊 反馈分析: {analysis.total_feedbacks} 个反馈")

        # 获取改进建议
        suggestions = await feedback_processor.get_improvement_suggestions()
        print(f"💡 改进建议: {len(suggestions)} 条")

        return True

    except Exception as e:
        print(f"❌ 反馈组件测试失败: {e}")
        return False


async def test_model_provider():
    """测试模型提供者"""
    print("\n" + "="*60)
    print("🤖 测试模型提供者")
    print("="*60)

    try:
        # 导入模型提供者模块
        from model_provider import ModelConfig, ModelType

        # 创建配置
        teacher_config = ModelConfig(
            model_name="deepseek-chat",
            model_type=ModelType.TEACHER,
            api_key="test-key",
            cost_per_token=0.001
        )

        student_config = ModelConfig(
            model_name="qwen2.5:14b",
            model_type=ModelType.STUDENT,
            api_base="http://localhost:11434",
            cost_per_token=0.0001
        )

        print(f"📋 教师模型配置: {teacher_config.model_name}")
        print(f"📋 学生模型配置: {student_config.model_name}")

        return True

    except Exception as e:
        print(f"❌ 模型提供者测试失败: {e}")
        return False


async def test_adaptation_component():
    """测试自适应组件"""
    print("\n" + "="*60)
    print("🎯 测试自适应组件")
    print("="*60)

    try:
        # 导入适应模块
        from adaptation import AdaptationConfig, AdaptationStrategy

        # 创建配置
        config = AdaptationConfig(
            strategy=AdaptationStrategy.MODERATE,
            adaptation_interval=50,
            confidence_threshold=0.6
        )

        print(f"📋 适应策略: {config.strategy.value}")
        print(f"📋 适应间隔: {config.adaptation_interval}")
        print(f"📋 置信度阈值: {config.confidence_threshold}")

        return True

    except Exception as e:
        print(f"❌ 自适应组件测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 玉珍健身 框架 组件简化测试")
    print("="*60)

    test_results = []

    # 测试各个组件
    test_results.append(("记忆组件", await test_memory_component()))
    test_results.append(("反馈组件", await test_feedback_component()))
    test_results.append(("模型提供者", await test_model_provider()))
    test_results.append(("自适应组件", await test_adaptation_component()))

    # 总结测试结果
    print("\n" + "="*60)
    print("🎉 测试完成!")
    print("="*60)

    print("✅ 测试结果:")
    passed = 0
    for component, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   - {component}: {status}")
        if result:
            passed += 1

    print(f"\n📊 总体结果: {passed}/{len(test_results)} 个组件通过测试")

    if passed == len(test_results):
        print("🎉 所有组件功能正常!")
    else:
        print("⚠️  部分组件需要进一步调试")


if __name__ == "__main__":
    print("玉珍健身 框架 简化组件测试")
    print("="*60)

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()