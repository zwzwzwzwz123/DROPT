@echo off
REM ========================================
REM 数据中心空调优化 - 快速启动脚本 (Windows)
REM ========================================

echo ========================================
echo 数据中心空调优化 - 快速启动
echo ========================================

REM 检查Python环境
echo.
echo [1/6] 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python未安装
    pause
    exit /b 1
)
python --version
echo [成功] Python已安装

REM 检查依赖
echo.
echo [2/6] 检查依赖包...
python -c "import torch" >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] PyTorch已安装
) else (
    echo [警告] PyTorch未安装，请运行: pip install torch
)

python -c "import tianshou" >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] Tianshou已安装
) else (
    echo [警告] Tianshou未安装，请运行: pip install tianshou
)

python -c "import gym" >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] Gym已安装
) else (
    echo [警告] Gym未安装，请运行: pip install gym
)

python -c "import numpy" >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] NumPy已安装
) else (
    echo [警告] NumPy未安装，请运行: pip install numpy
)

python -c "import pandas" >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] Pandas已安装
) else (
    echo [警告] Pandas未安装，请运行: pip install pandas
)

REM 创建必要目录
echo.
echo [3/6] 创建目录结构...
if not exist "data" mkdir data
if not exist "log_datacenter" mkdir log_datacenter
if not exist "scripts" mkdir scripts
echo [成功] 目录创建完成

REM 生成模拟数据
echo.
echo [4/6] 生成模拟数据...
if exist "data\weather_data.csv" if exist "data\workload_trace.csv" (
    echo [成功] 数据文件已存在，跳过生成
) else (
    echo 生成气象数据和负载轨迹（可能需要1-2分钟）...
    python scripts\generate_data.py
    if %errorlevel% equ 0 (
        echo [成功] 数据生成完成
    ) else (
        echo [错误] 数据生成失败
        pause
        exit /b 1
    )
)

REM 测试环境
echo.
echo [5/6] 测试环境...
echo 运行环境测试（可能需要2-3分钟）...
python scripts\test_datacenter_env.py
if %errorlevel% equ 0 (
    echo [成功] 环境测试通过
) else (
    echo [错误] 环境测试失败，请检查错误信息
    pause
    exit /b 1
)

REM 提供训练选项
echo.
echo [6/6] 选择训练模式...
echo.
echo 请选择训练模式：
echo   1) 快速演示（BC训练，1000轮，~5分钟）
echo   2) 标准训练（BC训练，50000轮，~1小时）
echo   3) 高性能训练（PG训练，200000轮，~6小时）
echo   4) 混合训练（BC+PG，~3小时）
echo   5) 自定义参数
echo   6) 跳过训练
echo.
set /p choice="请输入选项 (1-6): "

if "%choice%"=="1" (
    echo.
    echo [启动] 快速演示训练...
    python main_datacenter.py --bc-coef --expert-type pid --num-crac 4 --epoch 1000 --batch-size 128 --n-timesteps 3 --episode-length 50 --device cpu
) else if "%choice%"=="2" (
    echo.
    echo [启动] 标准训练...
    python main_datacenter.py --bc-coef --expert-type pid --num-crac 4 --epoch 50000 --batch-size 256 --n-timesteps 5 --device cuda:0
) else if "%choice%"=="3" (
    echo.
    echo [启动] 高性能训练...
    python main_datacenter.py --expert-type pid --num-crac 4 --epoch 200000 --batch-size 512 --n-timesteps 8 --gamma 0.99 --prioritized-replay --device cuda:0
) else if "%choice%"=="4" (
    echo.
    echo [启动] 混合训练（阶段1：BC预训练）...
    python main_datacenter.py --bc-coef --expert-type mpc --num-crac 4 --epoch 30000 --batch-size 256 --n-timesteps 6 --log-prefix pretrain --device cuda:0
    
    echo.
    echo [提示] 阶段2：PG微调
    echo 请手动运行以下命令，并替换^<MODEL_PATH^>为上一步保存的模型路径：
    echo.
    echo python main_datacenter.py --resume-path ^<MODEL_PATH^> --epoch 100000 --batch-size 512 --actor-lr 5e-5 --gamma 0.99 --device cuda:0
) else if "%choice%"=="5" (
    echo.
    echo [提示] 自定义训练
    echo 请手动运行 main_datacenter.py 并指定参数
    echo 示例：
    echo   python main_datacenter.py --help
    echo   python main_datacenter.py --bc-coef --epoch 10000 --device cuda:0
) else if "%choice%"=="6" (
    echo.
    echo [跳过] 跳过训练
) else (
    echo [错误] 无效选项
    pause
    exit /b 1
)

REM 完成
echo.
echo ========================================
echo [完成] 快速启动完成！
echo ========================================
echo.
echo 后续步骤：
echo   1. 查看训练日志: tensorboard --logdir=log_datacenter
echo   2. 测试模型: python main_datacenter.py --watch --resume-path ^<MODEL_PATH^>
echo   3. 阅读文档: type README_DATACENTER.md
echo   4. 查看迁移指南: type MIGRATION_GUIDE.md
echo.
echo 常用命令：
echo   # 查看帮助
echo   python main_datacenter.py --help
echo.
echo   # 快速训练
echo   python main_datacenter.py --bc-coef --epoch 50000
echo.
echo   # 启动TensorBoard
echo   tensorboard --logdir=log_datacenter
echo.
echo   # 测试环境
echo   python scripts\test_datacenter_env.py
echo.

pause

