#!/bin/bash
# ========================================
# 数据中心空调优化 - 快速启动脚本
# ========================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "数据中心空调优化 - 快速启动"
echo "========================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python环境
echo -e "\n${YELLOW}[1/6] 检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python已安装: $(python --version)${NC}"

# 检查依赖
echo -e "\n${YELLOW}[2/6] 检查依赖包...${NC}"
python -c "import torch" 2>/dev/null && echo -e "${GREEN}✓ PyTorch已安装${NC}" || echo -e "${RED}✗ PyTorch未安装，请运行: pip install torch${NC}"
python -c "import tianshou" 2>/dev/null && echo -e "${GREEN}✓ Tianshou已安装${NC}" || echo -e "${RED}✗ Tianshou未安装，请运行: pip install tianshou${NC}"
python -c "import gym" 2>/dev/null && echo -e "${GREEN}✓ Gym已安装${NC}" || echo -e "${RED}✗ Gym未安装，请运行: pip install gym${NC}"
python -c "import numpy" 2>/dev/null && echo -e "${GREEN}✓ NumPy已安装${NC}" || echo -e "${RED}✗ NumPy未安装，请运行: pip install numpy${NC}"
python -c "import pandas" 2>/dev/null && echo -e "${GREEN}✓ Pandas已安装${NC}" || echo -e "${RED}✗ Pandas未安装，请运行: pip install pandas${NC}"

# 创建必要目录
echo -e "\n${YELLOW}[3/6] 创建目录结构...${NC}"
mkdir -p data
mkdir -p log_datacenter
mkdir -p scripts
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 生成模拟数据
echo -e "\n${YELLOW}[4/6] 生成模拟数据...${NC}"
if [ -f "data/weather_data.csv" ] && [ -f "data/workload_trace.csv" ]; then
    echo -e "${GREEN}✓ 数据文件已存在，跳过生成${NC}"
else
    echo "生成气象数据和负载轨迹（可能需要1-2分钟）..."
    python scripts/generate_data.py
    echo -e "${GREEN}✓ 数据生成完成${NC}"
fi

# 测试环境
echo -e "\n${YELLOW}[5/6] 测试环境...${NC}"
echo "运行环境测试（可能需要2-3分钟）..."
python scripts/test_datacenter_env.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 环境测试通过${NC}"
else
    echo -e "${RED}✗ 环境测试失败，请检查错误信息${NC}"
    exit 1
fi

# 提供训练选项
echo -e "\n${YELLOW}[6/6] 选择训练模式...${NC}"
echo ""
echo "请选择训练模式："
echo "  1) 快速演示（BC训练，1000轮，~5分钟）"
echo "  2) 标准训练（BC训练，50000轮，~1小时）"
echo "  3) 高性能训练（PG训练，200000轮，~6小时）"
echo "  4) 混合训练（BC+PG，~3小时）"
echo "  5) 自定义参数"
echo "  6) 跳过训练"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo -e "\n${GREEN}启动快速演示训练...${NC}"
        python main_datacenter.py \
            --bc-coef \
            --expert-type pid \
            --num-crac 4 \
            --epoch 1000 \
            --batch-size 128 \
            --n-timesteps 3 \
            --episode-length 50 \
            --device cpu
        ;;
    2)
        echo -e "\n${GREEN}启动标准训练...${NC}"
        python main_datacenter.py \
            --bc-coef \
            --expert-type pid \
            --num-crac 4 \
            --epoch 50000 \
            --batch-size 256 \
            --n-timesteps 5 \
            --device cuda:0
        ;;
    3)
        echo -e "\n${GREEN}启动高性能训练...${NC}"
        python main_datacenter.py \
            --expert-type pid \
            --num-crac 4 \
            --epoch 200000 \
            --batch-size 512 \
            --n-timesteps 8 \
            --gamma 0.99 \
            --prioritized-replay \
            --device cuda:0
        ;;
    4)
        echo -e "\n${GREEN}启动混合训练（阶段1：BC预训练）...${NC}"
        python main_datacenter.py \
            --bc-coef \
            --expert-type mpc \
            --num-crac 4 \
            --epoch 30000 \
            --batch-size 256 \
            --n-timesteps 6 \
            --log-prefix pretrain \
            --device cuda:0
        
        echo -e "\n${GREEN}阶段2：PG微调...${NC}"
        echo "请手动运行以下命令，并替换<MODEL_PATH>为上一步保存的模型路径："
        echo ""
        echo "python main_datacenter.py \\"
        echo "    --resume-path <MODEL_PATH> \\"
        echo "    --epoch 100000 \\"
        echo "    --batch-size 512 \\"
        echo "    --actor-lr 5e-5 \\"
        echo "    --gamma 0.99 \\"
        echo "    --device cuda:0"
        ;;
    5)
        echo -e "\n${GREEN}自定义训练${NC}"
        echo "请手动运行 main_datacenter.py 并指定参数"
        echo "示例："
        echo "  python main_datacenter.py --help  # 查看所有参数"
        echo "  python main_datacenter.py --bc-coef --epoch 10000 --device cuda:0"
        ;;
    6)
        echo -e "\n${YELLOW}跳过训练${NC}"
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

# 完成
echo ""
echo "========================================"
echo -e "${GREEN}快速启动完成！${NC}"
echo "========================================"
echo ""
echo "后续步骤："
echo "  1. 查看训练日志: tensorboard --logdir=log_datacenter"
echo "  2. 测试模型: python main_datacenter.py --watch --resume-path <MODEL_PATH>"
echo "  3. 阅读文档: cat README_DATACENTER.md"
echo "  4. 查看迁移指南: cat MIGRATION_GUIDE.md"
echo ""
echo "常用命令："
echo "  # 查看帮助"
echo "  python main_datacenter.py --help"
echo ""
echo "  # 快速训练"
echo "  python main_datacenter.py --bc-coef --epoch 50000"
echo ""
echo "  # 启动TensorBoard"
echo "  tensorboard --logdir=log_datacenter"
echo ""
echo "  # 测试环境"
echo "  python scripts/test_datacenter_env.py"
echo ""

