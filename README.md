# VFBSOCPredictor

#### 介绍
VFBSOCPredictor是一个基于机器学习的钒液流电池(VFB)SOC(荷电状态)预测项目。该项目利用历史循环数据（如电压、电流、温度等参数），通过iTransformer时序模型训练和预测电池SOC状态，帮助优化电池管理和预测寿命。项目基于Python实现，支持数据预处理、模型训练、预测和可视化。

#### 软件架构
项目采用模块化结构：
- data_processing: 使用pandas处理Excel数据（如combined_SOC_data_20250604.xlsx）
- model: iTransformer模型基于PyTorch实现时序预测
- train_and_predict.py: 主脚本负责训练和预测，保存模型为.pth文件
- visualization: 使用matplotlib生成SOC比较图表（如SOC_Comparison.png）
- results/: 存储预测结果CSV和PNG文件

#### 安装教程

1. 克隆仓库：`git clone `
2. 进入目录：`cd vfbsocpredictor`
3. 确保安装了 UV 包管理器（https://docs.astral.sh/uv/getting-started/installation/）
4. 安装依赖：`uv sync`
5. 验证安装：`python -c "import pandas, torch; print('安装成功')"`

#### 使用说明

1. 准备数据：将循环数据放入root目录（如cycle79-80.xlsx）
2. 训练模型：python train_and_predict.py
3. 进行预测：python train_and_predict.py
4. 查看结果：results/文件夹中包含预测CSV文件和PNG对比图
5. 自定义：修改draw_picture.py调整可视化参数，或查看学习率曲线

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

