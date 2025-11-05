"""
初始化项目命令
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any

from .base import BaseCommand
from ..templates import get_domain_templates, create_project_from_template


class InitCommand(BaseCommand):
    """初始化新项目命令"""

    def get_help(self) -> str:
        return "初始化新的DAML-RAG项目"

    def get_description(self) -> str:
        return "创建一个新的DAML-RAG应用项目，支持多种领域模板"

    def add_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            'project_name',
            help='项目名称'
        )
        parser.add_argument(
            '--domain', '-d',
            choices=['fitness', 'healthcare', 'education', 'custom'],
            default='fitness',
            help='领域类型 (默认: fitness)'
        )
        parser.add_argument(
            '--template', '-t',
            help='自定义模板路径'
        )
        parser.add_argument(
            '--force', '-f',
            action='store_true',
            help='强制覆盖现有目录'
        )
        parser.add_argument(
            '--no-venv',
            action='store_true',
            help='不创建虚拟环境'
        )
        parser.add_argument(
            '--package-manager',
            choices=['pip', 'poetry', 'uv'],
            default='pip',
            help='包管理器 (默认: pip)'
        )

    async def execute(self, args: argparse.Namespace) -> int:
        """执行初始化命令"""
        project_name = args.project_name
        domain = args.domain

        # 验证项目名称
        if not self._validate_project_name(project_name):
            print(f"❌ 项目名称 '{project_name}' 无效")
            print("项目名称只能包含字母、数字、下划线和连字符，且不能以数字开头")
            return 1

        # 检查目录是否存在
        project_path = Path.cwd() / project_name
        if project_path.exists():
            if args.force:
                print(f"⚠️  目录 '{project_name}' 已存在，将被覆盖")
                shutil.rmtree(project_path)
            else:
                print(f"❌ 目录 '{project_name}' 已存在")
                print("使用 --force 强制覆盖")
                return 1

        try:
            print(f"🚀 创建DAML-RAG项目: {project_name}")
            print(f"📦 领域类型: {domain}")

            # 获取模板
            if args.template:
                template_path = Path(args.template)
                if not template_path.exists():
                    print(f"❌ 模板路径不存在: {template_path}")
                    return 1
                template_data = self._load_custom_template(template_path)
            else:
                template_data = get_domain_templates(domain)

            # 创建项目
            success = await create_project_from_template(
                project_path=project_path,
                project_name=project_name,
                template_data=template_data,
                domain=domain,
                package_manager=args.package_manager,
                create_venv=not args.no_venv
            )

            if not success:
                print("❌ 项目创建失败")
                return 1

            # 显示后续步骤
            self._show_next_steps(project_name, domain, not args.no_venv, args.package_manager)

            print("✅ 项目创建完成!")
            return 0

        except Exception as e:
            print(f"❌ 项目创建失败: {str(e)}")
            return 1

    def _validate_project_name(self, name: str) -> bool:
        """验证项目名称"""
        import re
        if not name:
            return False
        if name[0].isdigit():
            return False
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name))

    def _load_custom_template(self, template_path: Path) -> Dict[str, Any]:
        """加载自定义模板"""
        # 这里应该实现自定义模板加载逻辑
        # 暂时返回基础模板
        return get_domain_templates('custom')

    def _show_next_steps(self, project_name: str, domain: str, has_venv: bool, package_manager: str):
        """显示后续步骤"""
        print("\n🎯 后续步骤:")
        print(f"1. cd {project_name}")

        if has_venv:
            if package_manager == 'poetry':
                print("2. poetry install")
            elif package_manager == 'uv':
                print("2. uv sync")
            else:
                print("2. pip install -r requirements.txt")

        print("3. daml-rag dev")
        print("4. 打开浏览器访问 http://localhost:8000")

        print(f"\n📚 领域特定帮助:")
        if domain == 'fitness':
            print("- 健身领域工具: 23个专业健身工具")
            print("- 知识图谱: 2,447个健身实体节点")
            print("- 示例查询: '我想制定增肌计划'")
        elif domain == 'healthcare':
            print("- 医疗领域工具: 诊断、治疗、预防工具")
            print("- 知识图谱: 疾病、症状、药物关系")
            print("- 示例查询: '头痛的可能原因'")
        elif domain == 'education':
            print("- 教育领域工具: 课程设计、评估工具")
            print("- 知识图谱: 学科、概念、技能关系")
            print("- 示例查询: '设计Python入门课程'")

        print(f"\n🔗 更多信息:")
        print(f"- 项目文档: https://docs.daml-rag.org")
        print(f"- GitHub: https://github.com/daml-rag/daml-rag-framework")