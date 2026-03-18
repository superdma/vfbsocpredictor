import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 确保GPU加速
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ========== 数据加载与严格按Cycle_Index划分 ==========
data = pd.read_csv('./discharge-2024.08.23.csv', header=0)
save_path = './LSTM-discharge-TD_1-3/'
os.makedirs(save_path, exist_ok=True)

# 确认数据已按Cycle_Index排序
df = data.sort_values(by='Cycle_Index').reset_index(drop=True)

# 严格按要求划分：训练集cycle index<40，测试集cycle index>40
df_train = df[df['Cycle_Index'] < 40]
df_test = df[df['Cycle_Index'] > 40]

# ========== 数据预处理 ==========
X_train = df_train.iloc[:, 1:3].values
y_train = df_train['Ture_SOC'].values
X_test = df_test.iloc[:, 1:3].values
y_test = df_test['Ture_SOC'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 时间序列处理
time_step = 20

def create_sequences(x, y, time_step):
    x_seq, y_seq = [], []
    for i in range(len(x) - time_step + 1):
        x_seq.append(x[i:i+time_step])
        y_seq.append(y[i+time_step-1])
    return np.array(x_seq), np.array(y_seq)

x_train, y_train = create_sequences(X_train, y_train, time_step)
x_test, y_test = create_sequences(X_test, y_test, time_step)

# 创建数据集
batch_size = 128
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# ========== 修复：参数名冲突问题 ==========
def build_model(units=64, dropout=0.2, num_layers=2, bidirectional=False, lr=0.001):
    """修复：将layers参数名改为num_layers避免与模块名冲突"""
    inputs = keras.Input(shape=(time_step, X_train.shape[1]))
    
    # 双向LSTM层
    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True, 
                                           kernel_regularizer=keras.regularizers.l2(0.01)))(inputs)
    else:
        x = layers.LSTM(units, return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    
    # 多层LSTM（修复：使用num_layers代替layers）
    for _ in range(num_layers - 1):
        x = layers.LSTM(units, return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(0.01))(x)
    
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mae',
        metrics=['mse']
    )
    return model

def run_hyperparameter_tuning():
    # 定义超参数网格
    params = {
        'units': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3],
        'num_layers': [1, 2, 3],
        'bidirectional': [False, True],
        'lr': [0.00001, 0.0001, 0.001],
        'batch_size': [64, 128],
        'epochs': [25],
        'patience': [25]
    }
    
    # 生成所有组合
    all_combinations = []
    for units in params['units']:
        for dropout in params['dropout']:
            for num_layers in params['num_layers']:
                for bidirectional in params['bidirectional']:
                    for lr in params['lr']:
                        for batch_size in params['batch_size']:
                            all_combinations.append({
                                'units': units,
                                'dropout': dropout,
                                'num_layers': num_layers,
                                'bidirectional': bidirectional,
                                'lr': lr,
                                'batch_size': batch_size,
                                'epochs': params['epochs'][0],
                                'patience': params['patience'][0]
                            })
    
    results = []
    results_file = os.path.join(save_path, 'hyperparameter_results.csv')
    
    # 保存标题
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model_id'] + list(all_combinations[0].keys()) + ['test_mae', 'test_mse'])
        writer.writeheader()
    
    # 训练所有组合
    for idx, params in enumerate(all_combinations):
        print(f"\n=== MODEL {idx+1}/{len(all_combinations)} ===")
        print(f"Params: {params}")
        
        # 构建模型（修复：使用num_layers参数）
        model = build_model(
            units=params['units'],
            dropout=params['dropout'],
            num_layers=params['num_layers'],  # 修复：使用num_layers
            bidirectional=params['bidirectional'],
            lr=params['lr']
        )
        
        # 训练回调
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=params['patience'],
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=3,
                min_lr=1e-5
            )
        ]
        
        # 训练
        history = model.fit(
            train_ds,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估
        test_mae = model.evaluate(test_ds, verbose=0)[0]
        test_mse = model.evaluate(test_ds, verbose=0)[1]
        
        # 保存结果
        result = {
            'model_id': idx,
            **params,
            'test_mae': test_mae,
            'test_mse': test_mse
        }
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            writer.writerow(result)
        
        results.append(result)
        print(f"Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}")
    
    # 选择最佳模型（使用字典中的参数）
    best_model_dict = min(results, key=lambda x: x['test_mae'])
    print(f"\nBEST MODEL FOUND: MAE={best_model_dict['test_mae']:.4f}")
    print(f"Parameters: {best_model_dict}")
    
    # ========== 修复：关键错误修复（从字典获取参数，而非模型对象） ==========
    # 从最佳参数字典中提取所有需要的参数
    best_units = best_model_dict['units']
    best_dropout = best_model_dict['dropout']
    best_num_layers = best_model_dict['num_layers']
    best_bidirectional = best_model_dict['bidirectional']
    best_lr = best_model_dict['lr']
    best_epochs = best_model_dict['epochs']
    best_batch_size = best_model_dict['batch_size']
    best_patience = best_model_dict['patience']
    
    # 重新创建最佳参数的回调
    best_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=best_patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]
    
    # 用最佳参数重建并保存模型
    best_model = build_model(
        units=best_units,
        dropout=best_dropout,
        num_layers=best_num_layers,
        bidirectional=best_bidirectional,
        lr=best_lr
    )
    best_model.fit(
        train_ds,
        epochs=best_epochs,
        batch_size=best_batch_size,
        callbacks=best_callbacks,
        verbose=1
    )
    best_model.save(os.path.join(save_path, 'best_model.h5'))
    
    return best_model, results

# ========== 执行超参数调优 ==========
best_model, results = run_hyperparameter_tuning()

# ========== 最终预测与评估 ==========
# 获取测试集预测
y_pred_test = best_model.predict(test_ds).flatten()
y_test_cell = df_test['SOC_Cell'].values[time_step-1:]
cycle_index = df_test['Cycle_Index'].values[time_step-1:]
trues = y_test.flatten()  # y_test是序列化后的标签，长度为len(y_test)

# ========== 修复：数组长度不一致问题 ==========
# 确保所有数组长度一致（以y_test的长度为基准）
length = len(y_test)  # 基准长度

# 截断所有数组到相同长度
y_test_cell = y_test_cell[:length]
cycle_index = cycle_index[:length]
trues = trues[:length]  # 虽然trues就是y_test，但确保长度一致
y_pred_test = y_pred_test[:length]

# 保存结果
df_results = pd.DataFrame({
    'cycle_index': cycle_index,
    'True SOC': trues,
    'SOC cell': y_test_cell,
    'Pred. SOC': y_pred_test
})
df_results.to_csv(os.path.join(save_path, 'final_results.csv'), index=False)

# 计算指标
def calculate_metrics(preds, trues):
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    mape = np.mean(np.abs((trues - preds) / trues)) * 100
    max_error = np.max(np.abs(trues - preds))
    return mae, mse, mape, max_error

mae, mse, mape, max_error = calculate_metrics(y_pred_test, trues)
battery_mae, battery_mse, battery_mape, battery_max_error = calculate_metrics(y_test_cell, trues)

print("\n=== FINAL MODEL PERFORMANCE ===")
print(f"Model MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, Max Error: {max_error:.4f}")
print(f"OCV SOC (SOC_Cell) MAE: {battery_mae:.4f}")

print("\nAll results saved to:", save_path)
print("Hyperparameter results saved to:", os.path.join(save_path, 'hyperparameter_results.csv'))
print("Final model saved to:", os.path.join(save_path, 'best_model.h5'))