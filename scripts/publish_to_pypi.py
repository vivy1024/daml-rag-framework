#!/usr/bin/env python3
"""
PyPI发布脚本

使用方法:
    python scripts/publish_to_pypi.py [--test]

参数:
    --test: 发布到Test PyPI而不是正式PyPI
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_info(message):
    print(f"{Colors.BLUE}[INFO]{Colors.END} {message}")


def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.END} {message}")


def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING]{Colors.END} {message}")


def print_error(message):
    print(f"{Colors.RED}[ERROR]{Colors.END} {message}")


def run_command(cmd, check=True):
    """运行命令并返回结果"""
    print_info(f"执行命令: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0 and check:
        print_error(f"命令执行失败: {result.stderr}")
        sys.exit(1)
    
    return result


def check_dependencies():
    """检查必要的依赖"""
    print_info("检查依赖...")
    
    try:
        import build
        import twine
        print_success("依赖检查通过")
    except ImportError as e:
        print_error(f"缺少依赖: {e}")
        print_info("安装依赖: pip install build twine")
        sys.exit(1)


def clean_build():
    """清理旧的构建文件"""
    print_info("清理旧的构建文件...")
    
    dirs_to_clean = ['dist', 'build', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print_info(f"删除目录: {path}")
            else:
                path.unlink()
                print_info(f"删除文件: {path}")
    
    print_success("清理完成")


def build_package():
    """构建包"""
    print_info("构建包...")
    
    result = run_command("python -m build")
    
    if result.returncode == 0:
        print_success("包构建成功")
        
        # 列出构建的文件
        dist_files = list(Path('dist').glob('*'))
        print_info(f"构建文件: {len(dist_files)}个")
        for f in dist_files:
            print_info(f"  - {f.name}")
    else:
        print_error("包构建失败")
        sys.exit(1)


def check_package():
    """检查包的完整性"""
    print_info("检查包完整性...")
    
    result = run_command("twine check dist/*")
    
    if result.returncode == 0:
        print_success("包检查通过")
    else:
        print_error("包检查失败")
        sys.exit(1)


def upload_package(test_mode=False):
    """上传包到PyPI"""
    if test_mode:
        print_info("上传到Test PyPI...")
        repository = "testpypi"
        url = "https://test.pypi.org/simple/"
    else:
        print_info("上传到PyPI...")
        repository = "pypi"
        url = "https://pypi.org/simple/"
    
    # 检查是否有.pypirc配置
    pypirc_path = Path.home() / '.pypirc'
    if pypirc_path.exists():
        print_info("使用.pypirc配置")
        cmd = f"twine upload --repository {repository} dist/*"
    else:
        print_warning(".pypirc未找到，需要手动输入凭据")
        print_info("Username: __token__")
        print_info("Password: 你的PyPI API Token")
        
        if test_mode:
            cmd = "twine upload --repository-url https://test.pypi.org/legacy/ dist/*"
        else:
            cmd = "twine upload dist/*"
    
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print_success(f"上传成功！")
        print_info(f"查看包: {url}daml-rag-framework/")
    else:
        print_error("上传失败")
        print_error(result.stderr)
        sys.exit(1)


def verify_installation(test_mode=False):
    """验证安装"""
    print_info("验证安装...")
    
    if test_mode:
        cmd = "pip install --index-url https://test.pypi.org/simple/ --no-deps daml-rag-framework"
    else:
        cmd = "pip install --upgrade daml-rag-framework"
    
    print_info(f"运行: {cmd}")
    print_warning("请在新的虚拟环境中测试安装")


def main():
    """主函数"""
    # 检查参数
    test_mode = '--test' in sys.argv
    
    print_info("=" * 60)
    if test_mode:
        print_info("PyPI发布脚本 - Test PyPI模式")
    else:
        print_info("PyPI发布脚本 - 正式PyPI模式")
    print_info("=" * 60)
    
    # 确认发布
    if not test_mode:
        print_warning("你即将发布到正式PyPI！")
        confirm = input("确认继续？(yes/no): ")
        if confirm.lower() != 'yes':
            print_info("取消发布")
            sys.exit(0)
    
    # 执行发布流程
    try:
        check_dependencies()
        clean_build()
        build_package()
        check_package()
        upload_package(test_mode)
        verify_installation(test_mode)
        
        print_success("=" * 60)
        print_success("发布完成！")
        print_success("=" * 60)
        
    except KeyboardInterrupt:
        print_warning("\n发布已取消")
        sys.exit(1)
    except Exception as e:
        print_error(f"发布失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
