"""
DAML-RAG CLI 主入口
"""

import argparse
import sys
import asyncio
from typing import List, Optional

from .commands import (
    InitCommand,
    ScaffoldCommand,
    DeployCommand,
    HealthCheckCommand,
    ConfigCommand,
)
from .utils import setup_cli_logging


class DAMLRAGCLI:
    """DAML-RAG 命令行工具主类"""

    def __init__(self):
        self.commands = {
            'init': InitCommand(),
            'scaffold': ScaffoldCommand(),
            'deploy': DeployCommand(),
            'health': HealthCheckCommand(),
            'config': ConfigCommand(),
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            prog='daml-rag',
            description='DAML-RAG Framework - 面向垂直领域的自适应多源学习型RAG框架',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  daml-rag init my-fitness-app --domain fitness    # 创建健身应用
  daml-rag scaffold exercise-tool                   # 创建工具脚手架
  daml-rag deploy --platform docker                # Docker部署
  daml-rag health                                  # 健康检查
  daml-rag config show                              # 显示配置

更多信息请访问: https://github.com/daml-rag/daml-rag-framework
            """
        )

        # 全局选项
        parser.add_argument(
            '--version',
            action='version',
            version='%(prog)s 1.0.0'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='详细输出'
        )
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='静默模式'
        )
        parser.add_argument(
            '--config',
            help='配置文件路径',
            default='daml-rag.yaml'
        )

        # 子命令
        subparsers = parser.add_subparsers(
            dest='command',
            help='可用命令',
            metavar='COMMAND'
        )

        # 为每个命令创建子解析器
        for command_name, command in self.commands.items():
            command_parser = subparsers.add_parser(
                command_name,
                help=command.get_help(),
                description=command.get_description()
            )
            command.add_arguments(command_parser)

        return parser

    async def run(self, args: Optional[List[str]] = None) -> int:
        """运行CLI"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        # 设置日志
        setup_cli_logging(parsed_args.verbose, parsed_args.quiet)

        # 检查是否指定了命令
        if not parsed_args.command:
            parser.print_help()
            return 1

        # 执行命令
        try:
            command = self.commands.get(parsed_args.command)
            if not command:
                print(f"错误: 未知命令 '{parsed_args.command}'")
                return 1

            result = await command.execute(parsed_args)
            return result if isinstance(result, int) else 0

        except KeyboardInterrupt:
            print("\n操作被用户中断")
            return 1
        except Exception as e:
            print(f"错误: {str(e)}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main() -> int:
    """CLI主入口函数"""
    cli = DAMLRAGCLI()
    return asyncio.run(cli.run())


if __name__ == '__main__':
    sys.exit(main())