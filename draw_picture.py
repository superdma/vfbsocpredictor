import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_soc_example(result_df):
    """加载模型并执行预测示例"""
    try:
    
        # 将结果与原始数据合并
        df = result_df
        df['Difference'] = df['True SOC'] - df['Predict SOC']

        # 打印差异较大的数据
        threshold = 0.05
        large_diff_rows = df[np.abs(df['Difference']) > threshold]
        small_diff_rows = df[np.abs(df['Difference']) <= threshold]

        if not large_diff_rows.empty:
            print(f"\n以下行预测值与真实值相差大于 {threshold}:")
            print(large_diff_rows)
            print(large_diff_rows.info())
        else:
            print(f"\n所有预测值与真实值相差均小于等于 {threshold}")

        # 保存结果
        df.to_excel("预测结果.xlsx", index=False)
        print("\n结果已保存到预测结果.xlsx")

        # 创建带子图的画布 - 确保共享横轴
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)  # 关键：确保共享x轴

        # 主图：真实值 vs 预测值 - 优化配色方案
        # 使用差异色映射从深蓝到亮黄，增强对比度
        scatter = ax1.scatter(df.index, df['True SOC'], 
                      c=np.abs(df['Difference']),
                      cmap='coolwarm',  # 蓝-红冷暖对比色
                      s=45,
                      alpha=0.3,  # 提高透明度，使重叠点更易分辨
                      linewidths=0.5,  # 稍微增加线条宽度
                      edgecolors='royalblue',  # 白色轮廓增强清晰度
                      label='True SOC',
                      zorder=3)

        # 预测值 - 使用更高对比度的颜色
        ax1.plot(df.index, df['Predict SOC'], 
                color='#FF4500',  # 亮橙色，与蓝色形成互补
                linewidth=1.5, 
                alpha=0.95,
                label='Predict SOC',
                zorder=2)

        # 添加颜色条并设置位置 - 使用专用坐标轴防止主图变形
        # 在图形上创建新的坐标轴用于colorbar
        cbar_ax = fig.add_axes([0.92, 0.32, 0.0125, 0.60])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Absolute Error', fontsize=10, labelpad=8)

        # 标记差异大于阈值的点
        if not large_diff_rows.empty:
            # 加大标记尺寸，使用空心圆
            ax1.scatter(large_diff_rows.index, large_diff_rows['Predict SOC'], 
                        facecolors='none',
                        edgecolors='#8B0000',  # 深红色边缘
                        s=110, 
                        marker='o',
                        linewidths=1.8,
                        alpha=1.0,
                        label=f'Error > {threshold}',
                        zorder=5)
            
            # 误差连接线改为渐变宽度箭头
            for idx, row in large_diff_rows.iterrows():
                ax1.annotate("", 
                            xy=(idx, row['True SOC']), 
                            xytext=(idx, row['Predict SOC']),
                            arrowprops=dict(arrowstyle="fancy", 
                                            color='#D2691E',  # 巧克力色
                                            alpha=0.7,
                                            linewidth=1.0,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle="arc3,rad=0.2"),
                            zorder=4)

        # 设置主图格式 - 增加标题字体
        ax1.set_title('True SOC vs Predicted SOC', fontsize=15, fontweight='bold', pad=12)
        ax1.set_ylabel('SOC Value', fontsize=12, labelpad=10)
        ax1.legend(loc='upper right', fontsize=10)  # 移至右上角避免与图中部分重叠
        ax1.grid(True, linestyle='--', alpha=0.5, color='lightgray')

        # 子图2：差异分布 - 保持与主图x轴对齐
        # 填充区使用半透明蓝色和红色
        ax2.fill_between(df.index, df['Difference'], 0,
                        where=df['Difference'] >= 0,
                        color='royalblue', alpha=0.6, interpolate=True)
        ax2.fill_between(df.index, df['Difference'], 0,
                        where=df['Difference'] < 0,
                        color='crimson', alpha=0.6, interpolate=True)

        ax2.axhline(y=0, color='dimgray', linestyle='-', linewidth=1.2)

        # 阈值范围使用浅灰色背景
        ax2.axhspan(-threshold, threshold, facecolor='lightgrey', alpha=0.4)

        # 添加阈值边界线
        ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.8, linewidth=1)
        ax2.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.8, linewidth=1)

        # 设置差异图格式
        ax2.set_title('Prediction Error Distribution', fontsize=13, pad=10)
        ax2.set_ylabel('Error', fontsize=11, labelpad=8)
        ax2.set_xlabel('Sample Index', fontsize=12, labelpad=10)
        ax2.grid(True, linestyle=':', alpha=0.4, color='gainsboro')

        # 误差统计信息框
        mae = np.abs(df['Difference']).mean()
        max_err = df['Difference'].abs().max()
        ax2.text(0.97, 0.92, 
                f"MAE: {mae:.4f}\nMax Error: {max_err:.4f}", 
                transform=ax2.transAxes, 
                ha='right', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # 确保横轴刻度标签可见 - 重要对齐步骤
        plt.setp(ax1.get_xticklabels(), visible=False)  # 主图隐藏x轴标签
        ax2.tick_params(axis='x', which='both', labelsize=9)  # 子图设置刻度字号
        fig.align_ylabels()  # 对齐y轴标签

        # 调整子图布局以确保对齐
        plt.subplots_adjust(left=0.08, right=0.91, bottom=0.08, top=0.92, hspace=0.1)

        # 保存高清图像
        plt.savefig("SOC_Comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e.filename}")
    except Exception as e:
        print(f"预测发生异常: {str(e)}")

def merge_and_save_files():
    # 读取第一个CSV文件
    df1 = pd.read_csv('./results/Cycle_79_test_预测结果.csv')
    
    # 添加一个标识来源的列（可选）
    df1['Cycle Index'] = 79
    
    # 读取第二个CSV文件
    df2 = pd.read_csv('./results/Cycle_80_test_预测结果.csv')
    
    # 添加一个标识来源的列（可选）
    df2['Cycle Index'] = 80
    
    # 合并两个DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 重置索引
    merged_df.reset_index(drop=True, inplace=True)
    
    # 添加一个新的连续索引列（如果原始索引没有保留）
    merged_df.insert(0, 'Index', range(1, len(merged_df) + 1))
    
    # 保存结果到新文件

    # 打印统计信息
    print(f"合并完成，总行数: {merged_df.shape}")
    print(f"Cycle 79行数: {len(df1)}")
    print(f"Cycle 80行数: {len(df2)}")

    merged_df = merged_df[['Index', '真实SOC', '预测SOC', 'Cycle Index']]
    merged_df.columns = ['Index', 'True SOC', 'Predict SOC', 'Cycle Index']
    draw_soc_example(merged_df)

if __name__ == "__main__":
    merge_and_save_files()