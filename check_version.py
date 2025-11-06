#!/usr/bin/env python3
"""
DAML-RAG Framework - 版本检查脚本
用于验证框架安装和版本信息
"""

def check_version():
    """检查 DAML-RAG 框架版本和安装状态"""
    
    print("DAML-RAG Framework - Version Check")
    print("=" * 50)
    print()
    
    try:
        # 导入主包
        import daml_rag
        print(f"[OK] Framework Version: {daml_rag.__version__}")
        print(f"[OK] Author: {daml_rag.__author__}")
        print(f"[OK] License: {daml_rag.__license__}")
        print()
        
        # 检查核心模块
        print("Core Modules Check:")
        
        try:
            from daml_rag import DAMLRAGFramework
            print("  [OK] DAMLRAGFramework")
        except ImportError as e:
            print(f"  [FAIL] DAMLRAGFramework: {e}")
        
        try:
            from daml_rag.retrieval import VectorRetriever
            print("  [OK] retrieval module")
        except ImportError as e:
            print(f"  [FAIL] retrieval module: {e}")
        
        try:
            from daml_rag.learning import ModelProvider
            print("  [OK] learning module")
        except ImportError as e:
            print(f"  [FAIL] learning module: {e}")
        
        try:
            from daml_rag.orchestration import Orchestrator
            print("  [OK] orchestration module")
        except ImportError as e:
            print(f"  [FAIL] orchestration module: {e}")
        
        try:
            from daml_rag.adapters import FitnessDomainAdapter
            print("  [OK] adapters module")
        except ImportError as e:
            print(f"  [FAIL] adapters module: {e}")
        
        try:
            from daml_rag.cli import main
            print("  [OK] cli module")
        except ImportError as e:
            print(f"  [FAIL] cli module: {e}")
        
        print()
        print("All core modules checked!")
        print()
        print("Next steps:")
        print("  1. Quick start: cat QUICKSTART.md")
        print("  2. View examples: ls examples/")
        print("  3. Run CLI: daml-rag --help")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Framework not installed: {e}")
        print()
        print("Try:")
        print("  1. Install: pip install -e .")
        print("  2. Build: ./scripts/build.sh")
        print("  3. Docs: cat BUILD_AND_PUBLISH.md")
        return False


if __name__ == "__main__":
    import sys
    success = check_version()
    sys.exit(0 if success else 1)

