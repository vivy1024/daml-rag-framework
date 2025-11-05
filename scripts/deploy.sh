#!/bin/bash
# DAML-RAG框架部署脚本
# 一键部署三层检索系统和MCP服务器

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker和Docker Compose
check_dependencies() {
    log_info "检查依赖..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi

    log_success "依赖检查通过"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."

    mkdir -p data/qdrant
    mkdir -p data/neo4j
    mkdir -p data/redis
    mkdir -p logs
    mkdir -p config

    log_success "目录创建完成"
}

# 构建和启动服务
start_services() {
    log_info "启动DAML-RAG框架服务..."

    # 进入docker目录
    cd docker

    # 构建并启动所有服务
    docker-compose up -d --build

    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."

    # 等待Qdrant
    log_info "等待Qdrant服务..."
    until curl -f http://localhost:6333/health &>/dev/null; do
        echo -n "."
        sleep 2
    done
    echo -e "\n✅ Qdrant服务就绪"

    # 等待Neo4j
    log_info "等待Neo4j服务..."
    until curl -f http://localhost:7474 &>/dev/null; do
        echo -n "."
        sleep 2
    done
    echo -e "\n✅ Neo4j服务就绪"

    # 等待MCP服务器
    log_info "等待MCP服务器..."
    until curl -f http://localhost:8002/health &>/dev/null; do
        echo -n "."
        sleep 2
    done
    echo -e "\n✅ MCP服务器就绪"

    log_success "所有服务已就绪"
}

# 初始化数据
initialize_data() {
    log_info "初始化数据..."

    # 这里可以添加数据初始化脚本
    # 例如：导入知识图谱数据、向量数据等

    log_success "数据初始化完成"
}

# 显示服务状态
show_status() {
    log_info "服务状态："
    echo ""

    cd docker
    docker-compose ps

    echo ""
    log_info "服务访问地址："
    echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
    echo "  • Neo4j Browser: http://localhost:7474 (neo4j/build_body_2024)"
    echo "  • MCP服务器: http://localhost:8002"
    echo "  • MCP API文档: http://localhost:8002/docs"
    echo "  • Redis: localhost:6379"
    echo ""
}

# 运行健康检查
health_check() {
    log_info "执行健康检查..."

    # 检查MCP服务器
    if curl -f http://localhost:8002/health &>/dev/null; then
        log_success "MCP服务器健康"
    else
        log_error "MCP服务器不健康"
        return 1
    fi

    # 检查Qdrant
    if curl -f http://localhost:6333/health &>/dev/null; then
        log_success "Qdrant服务健康"
    else
        log_error "Qdrant服务不健康"
        return 1
    fi

    # 检查Neo4j
    if curl -f http://localhost:7474 &>/dev/null; then
        log_success "Neo4j服务健康"
    else
        log_error "Neo4j服务不健康"
        return 1
    fi

    log_success "所有服务健康检查通过"
}

# 显示帮助信息
show_help() {
    echo "DAML-RAG框架部署脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动所有服务"
    echo "  stop      停止所有服务"
    echo "  restart   重启所有服务"
    echo "  status    显示服务状态"
    echo "  logs      显示服务日志"
    echo "  health    执行健康检查"
    echo "  clean     清理所有数据和服务"
    echo "  help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start      # 启动所有服务"
    echo "  $0 status     # 查看服务状态"
    echo "  $0 logs       # 查看日志"
    echo ""
}

# 清理服务
clean_services() {
    log_warning "这将删除所有服务数据和容器，是否继续？(y/N)"
    read -r response

    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "清理服务..."

        cd docker
        docker-compose down -v
        docker system prune -f

        log_success "清理完成"
    else
        log_info "清理已取消"
    fi
}

# 主函数
main() {
    case "${1:-start}" in
        start)
            check_dependencies
            create_directories
            start_services
            wait_for_services
            initialize_data
            show_status
            health_check
            ;;
        stop)
            log_info "停止所有服务..."
            cd docker
            docker-compose down
            log_success "服务已停止"
            ;;
        restart)
            log_info "重启所有服务..."
            cd docker
            docker-compose restart
            wait_for_services
            show_status
            health_check
            ;;
        status)
            show_status
            ;;
        logs)
            cd docker
            docker-compose logs -f
            ;;
        health)
            health_check
            ;;
        clean)
            clean_services
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"