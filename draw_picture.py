import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_soc_example(result_df):
    """Load the model and execute a prediction example"""
    try:
        # Merge the results with the original data
        df = result_df
        df['Difference'] = df['True SOC'] - df['Predict SOC']

        # Print rows where differences are larger
        threshold = 0.05
        large_diff_rows = df[np.abs(df['Difference']) > threshold]
        small_diff_rows = df[np.abs(df['Difference']) <= threshold]

        if not large_diff_rows.empty:
            print(f"\nThe following predictions have an absolute error greater than {threshold}:")
            print(large_diff_rows)
            print(large_diff_rows.info())
        else:
            print(f"\nAll predictions have an absolute error less than or equal to {threshold}")

        # Save the results
        df.to_excel("prediction_results.xlsx", index=False)
        print("\nResults have been saved to prediction_results.xlsx")

        # Create a figure with subplots - Ensure sharing of x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)  # Key: ensure shared x-axis

        # Main plot: True vs Predicted values - Optimized color scheme
        scatter = ax1.scatter(df.index, df['True SOC'], 
                      c=np.abs(df['Difference']),
                      cmap='coolwarm',  # Blue-Red cold-warm contrast colors
                      s=45,
                      alpha=0.3,  # Increase transparency for easier overlap identification
                      linewidths=0.5,  # Slightly increase line width
                      edgecolors='royalblue',  # White outline enhances clarity
                      label='True SOC',
                      zorder=3)

        # Prediction value - Use a higher contrast color
        ax1.plot(df.index, df['Predict SOC'], 
                color='#FF4500',  # Bright orange, complementary to blue
                linewidth=1.5, 
                alpha=0.95,
                label='Predict SOC',
                zorder=2)

        # Add a color bar and set its position - Use dedicated axis to prevent distortion of main plot
        cbar_ax = fig.add_axes([0.92, 0.32, 0.0125, 0.60])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Absolute Error', fontsize=10, labelpad=8)

        # Highlight points where the difference exceeds the threshold
        if not large_diff_rows.empty:
            ax1.scatter(large_diff_rows.index, large_diff_rows['Predict SOC'], 
                        facecolors='none',
                        edgecolors='#8B0000',  # Dark red edges
                        s=110, 
                        marker='o',
                        linewidths=1.8,
                        alpha=1.0,
                        label=f'Error > {threshold}',
                        zorder=5)
            
            # Change error connection lines to gradient width arrows
            for idx, row in large_diff_rows.iterrows():
                ax1.annotate("", 
                            xy=(idx, row['True SOC']), 
                            xytext=(idx, row['Predict SOC']),
                            arrowprops=dict(arrowstyle="fancy", 
                                            color='#D2691E',  # Chocolate color
                                            alpha=0.7,
                                            linewidth=1.0,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle="arc3,rad=0.2"),
                            zorder=4)

        # Set main plot format - Increase title font size
        ax1.set_title('True SOC vs Predicted SOC', fontsize=15, fontweight='bold', pad=12)
        ax1.set_ylabel('SOC Value', fontsize=12, labelpad=10)
        ax1.legend(loc='upper right', fontsize=10)  # Move to top-right corner to avoid overlapping with parts of the chart
        ax1.grid(True, linestyle='--', alpha=0.5, color='lightgray')

        # Subplot 2: Distribution of differences - Keep aligned with the main plot's x-axis
        ax2.fill_between(df.index, df['Difference'], 0,
                        where=df['Difference'] >= 0,
                        color='royalblue', alpha=0.6, interpolate=True)
        ax2.fill_between(df.index, df['Difference'], 0,
                        where=df['Difference'] < 0,
                        color='crimson', alpha=0.6, interpolate=True)

        ax2.axhline(y=0, color='dimgray', linestyle='-', linewidth=1.2)

        # Threshold range uses light gray background
        ax2.axhspan(-threshold, threshold, facecolor='lightgrey', alpha=0.4)

        # Add threshold boundary lines
        ax2.axhline(y=threshold, color='gray', linestyle='--', alpha=0.8, linewidth=1)
        ax2.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.8, linewidth=1)

        # Set format for the difference plot
        ax2.set_title('Prediction Error Distribution', fontsize=13, pad=10)
        ax2.set_ylabel('Error', fontsize=11, labelpad=8)
        ax2.set_xlabel('Sample Index', fontsize=12, labelpad=10)
        ax2.grid(True, linestyle=':', alpha=0.4, color='gainsboro')

        # Statistics information box for errors
        mae = np.abs(df['Difference']).mean()
        max_err = df['Difference'].abs().max()
        ax2.text(0.97, 0.92, 
                f"MAE: {mae:.4f}\nMax Error: {max_err:.4f}", 
                transform=ax2.transAxes, 
                ha='right', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # Ensure x-axis tick labels visibility - Important alignment step
        plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x-axis labels on the main plot
        ax2.tick_params(axis='x', which='both', labelsize=9)  # Set tick label font size on subplot
        fig.align_ylabels()  # Align y-axis labels

        # Adjust subplot layout to ensure alignment
        plt.subplots_adjust(left=0.08, right=0.91, bottom=0.08, top=0.92, hspace=0.1)

        # Save high-resolution image
        plt.savefig("SOC_Comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except Exception as e:
        print(f"An exception occurred during prediction: {str(e)}")


def merge_and_save_files():
    # Read the first CSV file
    df1 = pd.read_csv('./results/Cycle_79_test_预测结果.csv')
    
    # Optionally add a column identifying the source
    df1['Cycle Index'] = 79
    
    # Read the second CSV file
    df2 = pd.read_csv('./results/Cycle_80_test_预测结果.csv')
    
    # Optionally add a column identifying the source
    df2['Cycle Index'] = 80
    
    # Merge two DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Reset index
    merged_df.reset_index(drop=True, inplace=True)
    
    # Insert a new continuous index column if original indices are not retained
    merged_df.insert(0, 'Index', range(1, len(merged_df) + 1))
    
    # Print statistical information
    print(f"Merging completed, total number of rows: {merged_df.shape}")
    print(f"Number of rows in Cycle 79: {len(df1)}")
    print(f"Number of rows in Cycle 80: {len(df2)}")

    merged_df = merged_df[['Index', 'True SOC', 'Predict SOC', 'Cycle Index']]
    draw_soc_example(merged_df)


if __name__ == "__main__":
    merge_and_save_files()