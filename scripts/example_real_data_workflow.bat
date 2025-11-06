@echo off
REM ========================================
REM 真实数据集成完整工作流示例 (Windows)
REM ========================================
REM 从数据预处理到模型训练的完整流程

setlocal enabledelayedexpansion

echo ========================================
echo 真实数据集成工作流
echo ========================================

REM ========== 配置参数 ==========
set RAW_DATA=raw_data\datacenter_log.csv
set PROCESSED_DATA=data\real_data_processed.csv
set CALIBRATED_PARAMS=results\calibrated_params.json
set LOG_DIR=log_real_data

REM ========== Phase 1: 数据预处理 ==========
echo.
echo ========================================
echo Phase 1: 数据预处理
echo ========================================

if not exist "%RAW_DATA%" (
    echo 错误: 找不到原始数据文件 %RAW_DATA%
    echo 请将您的真实数据放置在该路径，或修改脚本中的RAW_DATA变量
    pause
    exit /b 1
)

echo 步骤 1.1: 数据清洗和验证...
python scripts\preprocess_real_data.py ^
    --input "%RAW_DATA%" ^
    --output "%PROCESSED_DATA%" ^
    --resample 5T ^
    --smooth 3 ^
    --validate ^
    --plot

if errorlevel 1 (
    echo 错误: 数据预处理失败
    pause
    exit /b 1
)

echo ✓ 数据预处理完成
echo   - 输出文件: %PROCESSED_DATA%
echo   - 质量报告: %PROCESSED_DATA:~0,-4%_quality_report.txt
echo   - 可视化: data\data_visualization.png

REM ========== Phase 2: 模型校准 ==========
echo.
echo ========================================
echo Phase 2: 模型参数校准
echo ========================================

echo 步骤 2.1: 使用贝叶斯优化校准参数...
python scripts\calibrate_model.py ^
    --real-data "%PROCESSED_DATA%" ^
    --method bayesian ^
    --num-crac 4 ^
    --output "%CALIBRATED_PARAMS%"

if errorlevel 1 (
    echo 错误: 模型校准失败
    pause
    exit /b 1
)

echo ✓ 模型校准完成
echo   - 校准参数: %CALIBRATED_PARAMS%

REM 显示校准结果
echo.
echo 校准参数:
type "%CALIBRATED_PARAMS%"

REM ========== Phase 3: 基线训练（纯仿真） ==========
echo.
echo ========================================
echo Phase 3: 基线训练（纯仿真）
echo ========================================

echo 步骤 3.1: 使用仿真数据训练基线模型...
python main_datacenter.py ^
    --bc-coef ^
    --epoch 30000 ^
    --num-crac 4 ^
    --calibrated-params "%CALIBRATED_PARAMS%" ^
    --logdir "%LOG_DIR%_baseline" ^
    --device cuda:0

if errorlevel 1 (
    echo 警告: 基线训练可能失败或被中断
)

echo ✓ 基线训练完成
echo   - 模型保存: %LOG_DIR%_baseline\policy_best.pth

REM ========== Phase 4: 混合训练 ==========
echo.
echo ========================================
echo Phase 4: 混合训练（仿真+真实）
echo ========================================

echo 步骤 4.1: 使用渐进式混合策略训练...
python main_datacenter.py ^
    --bc-coef ^
    --real-data "%PROCESSED_DATA%" ^
    --real-data-ratio-schedule progressive ^
    --data-augmentation ^
    --epoch 100000 ^
    --num-crac 4 ^
    --calibrated-params "%CALIBRATED_PARAMS%" ^
    --logdir "%LOG_DIR%_mixed" ^
    --device cuda:0

if errorlevel 1 (
    echo 警告: 混合训练可能失败或被中断
)

echo ✓ 混合训练完成
echo   - 模型保存: %LOG_DIR%_mixed\policy_best.pth

REM ========== Phase 5: 微调 ==========
echo.
echo ========================================
echo Phase 5: 真实数据微调
echo ========================================

echo 步骤 5.1: 使用纯真实数据微调...
python main_datacenter.py ^
    --real-data "%PROCESSED_DATA%" ^
    --real-data-ratio 1.0 ^
    --resume-path "%LOG_DIR%_mixed\policy_best.pth" ^
    --epoch 20000 ^
    --lr 1e-5 ^
    --num-crac 4 ^
    --calibrated-params "%CALIBRATED_PARAMS%" ^
    --logdir "%LOG_DIR%_finetuned" ^
    --device cuda:0

if errorlevel 1 (
    echo 警告: 微调可能失败或被中断
)

echo ✓ 微调完成
echo   - 模型保存: %LOG_DIR%_finetuned\policy_best.pth

REM ========== 完成 ==========
echo.
echo ========================================
echo 工作流完成！
echo ========================================
echo.
echo 训练结果:
echo   - 基线模型: %LOG_DIR%_baseline\policy_best.pth
echo   - 混合模型: %LOG_DIR%_mixed\policy_best.pth
echo   - 微调模型: %LOG_DIR%_finetuned\policy_best.pth
echo.
echo 查看训练曲线:
echo   tensorboard --logdir %LOG_DIR%_baseline
echo   tensorboard --logdir %LOG_DIR%_mixed
echo   tensorboard --logdir %LOG_DIR%_finetuned
echo.
echo 下一步:
echo   1. 查看TensorBoard了解训练过程
echo   2. 使用测试脚本评估模型性能
echo   3. 在真实环境中小规模部署测试
echo.

pause

