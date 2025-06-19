import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm import tqdm

# 设置中文显示
font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # 替换为实际路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 1. 定义Transformer模型架构
class iTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(iTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.BatchNorm1d(d_model),  # 添加BatchNorm
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model, output_dim)
        # )
    
    def forward(self, src):
        # 嵌入层
        src = self.embedding(src)
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码器
        output = self.transformer_encoder(src)
        
        # 取序列最后一个时间步的输出
        output = output[:, -1, :]
        
        # 全连接层
        output = self.fc(output)
        return output

# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 2. 数据集类
class CycleDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        """
        初始化数据集
        
        参数:
        data -- 包含所有循环数据的字典 {cycle_index: (features, targets)}
        sequence_length -- 输入序列长度
        """
        self.sequences = []
        self.targets = []
        
        for cycle_index, (features, target) in data.items():
            # 创建序列
            for i in range(len(features) - sequence_length):
                seq = features[i:i+sequence_length]
                label = target[i+sequence_length-1]  # 使用序列最后一个时间点的真实SOC
                
                self.sequences.append(seq)
                self.targets.append(label)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target

# 3. 数据处理函数
def load_and_prepare_data(file_path, selected_cycles=None):
    """
    加载数据并进行预处理
    
    参数:
    file_path -- Excel文件路径
    selected_cycles -- 选择的循环列表 (默认为None，加载所有循环)
    
    返回:
    cycle_data -- 字典 {cycle_index: (features, targets)}
    """
    # 加载数据
    df = pd.read_excel(file_path)
    print(f"数据加载完成，总行数: {len(df)}")
    
    # 确保列名正确
    column_mapping = {
        'Data Point': 'Data_Point',
        'Cycle Index': 'Cycle_Index',
        'Cycle_Index': 'Cycle_Index',
        'Current (A)': 'Current_A',
        'Current_A': 'Current_A',
        'Voltage (V)': 'Voltage_V',
        'Voltage_V': 'Voltage_V',
        'True SOC': 'True_SOC',
        'True_SOC': 'True_SOC'
    }
    
    # 重命名列
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # 删除不需要的列
    keep_columns = ['Cycle_Index', 'Current_A', 'Voltage_V', 'True_SOC']
    if 'Data_Point' in df.columns:
        keep_columns.append('Data_Point')
    else:
        # 添加行号作为替代排序依据
        df['Data_Point'] = df.index
    
    # 删除无关列
    drop_columns = [col for col in df.columns if col not in keep_columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    # 筛选指定的循环
    if selected_cycles:
        df = df[df['Cycle_Index'].isin(selected_cycles)]
    
    # 按循环分组
    grouped = df.groupby('Cycle_Index')
    
    # 准备数据字典
    cycle_data = {}
    
    # 按循环存储数据，确保每个循环内的顺序
    for cycle_index, group in grouped:
        # 如果没有Data_Point列，使用原始索引顺序
        group = group.sort_values('Data_Point')  # 确保数据点顺序

        # 提取特征和目标
        features = group[['Current_A', 'Voltage_V', 'Cycle_Index']].values
        targets = group['True_SOC'].values
        
        cycle_data[cycle_index] = (features, targets)
    
    print(f"数据处理完成，加载了 {len(cycle_data)} 个循环的数据")
    return cycle_data

# 4. 数据标准化
def standardize_data(train_data, test_data):
    """
    标准化训练和测试数据
    
    参数:
    train_data -- 训练数据字典
    test_data -- 测试数据字典
    
    返回:
    train_data_scaled -- 标准化后的训练数据
    test_data_scaled -- 标准化后的测试数据
    scaler -- 用于逆变换的标准化器
    """
    # 提取所有特征用于拟合标准化器
    all_features = []
    for features, _ in train_data.values():
        all_features.extend(features)
    
    # 创建标准化器
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # 标准化训练数据
    train_data_scaled = {}
    for cycle, (features, targets) in train_data.items():
        features_scaled = scaler.transform(features)
        train_data_scaled[cycle] = (features_scaled, targets)
    
    # 标准化测试数据
    test_data_scaled = {}
    for cycle, (features, targets) in test_data.items():
        features_scaled = scaler.transform(features)
        test_data_scaled[cycle] = (features_scaled, targets)
    
    return train_data_scaled, test_data_scaled, scaler

# 5. 模型训练函数 - 修改了学习率调度器的使用方式
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):
    """
    训练模型
    
    参数:
    model -- 要训练的模型
    train_loader -- 训练数据加载器
    val_loader -- 验证数据加载器
    epochs -- 训练轮数
    lr -- 学习率
    patience -- 提前停止的耐心值
    
    返回:
    train_losses -- 训练损失列表
    val_losses -- 验证损失列表
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # 添加L2正则化
    
    # 学习率调度器 - 移除了verbose参数
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=patience//2
    # )
    # 增加学习率调度器的耐心值
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=16
    )
    
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    lr_history = []  # 用于记录学习率变化
    
    for epoch in tqdm(range(epochs), desc="训练进度"):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * sequences.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                
                epoch_val_loss += loss.item() * sequences.size(0)
        
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step(epoch_val_loss)
        
        # 检查学习率是否发生变化
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            tqdm.write(f"学习率减小到 {new_lr:.7f} 在epoch {epoch+1}")
        
        # 保存当前学习率
        lr_history.append(new_lr)
        
        # 打印进度
        tqdm.write(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | LR: {new_lr:.6f}')
        
        # 早停检查
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    if best_model:
        model.load_state_dict(best_model)
    
    # 绘制学习率变化图
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history, 'b-o')
    plt.title('训练期间学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/学习率变化曲线.png', dpi=300)
    plt.close()
    
    return model, train_losses, val_losses

# 6. 预测函数
def predict(model, data_loader):
    """
    使用模型进行预测
    
    参数:
    model -- 训练好的模型
    data_loader -- 数据加载器
    
    返回:
    all_preds -- 所有预测值
    all_targets -- 所有真实值
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            outputs = model(sequences)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_preds).flatten(), np.array(all_targets).flatten()

# 7. 可视化函数
def plot_results(preds, targets, cycle_index, dataset_type):
    """
    可视化预测结果
    
    参数:
    preds -- 预测值数组
    targets -- 真实值数组
    cycle_index -- 循环索引
    dataset_type -- 数据集类型 ('train'或'test')
    """
    plt.figure(figsize=(14, 8))
    
    # 真实值与预测值对比
    plt.subplot(2, 2, (1, 2))
    plt.plot(targets, 'b-', alpha=0.7, label='真实SOC')
    plt.plot(preds, 'r--', alpha=0.9, label='预测SOC')
    plt.title(f'Cycle {cycle_index} {dataset_type} SOC预测对比')
    plt.xlabel('数据点索引')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)
    
    # 预测误差分布
    plt.subplot(2, 2, 3)
    errors = preds - targets
    sns.histplot(errors, kde=True)
    plt.title('预测误差分布')
    plt.xlabel('预测误差')
    plt.grid(True)
    
    # 误差散点图
    plt.subplot(2, 2, 4)
    plt.scatter(targets, preds, alpha=0.3)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.title('真实值 vs 预测值')
    plt.xlabel('真实SOC')
    plt.ylabel('预测SOC')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/Cycle_{cycle_index}_{dataset_type}_预测结果.png', dpi=300)
    plt.close()

# 8. 保存预测结果
def save_predictions(preds, targets, cycle_index, dataset_type):
    """
    保存预测结果到CSV文件
    
    参数:
    preds -- 预测值数组
    targets -- 真实值数组
    cycle_index -- 循环索引
    dataset_type -- 数据集类型 ('train'或'test')
    """
    results = pd.DataFrame({
        '数据点索引': np.arange(len(preds)),
        '预测SOC': preds,
        '真实SOC': targets
    })
    results.to_csv(f'results/Cycle_{cycle_index}_{dataset_type}_预测结果.csv', index=False)

# 9. 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义数据集路径
    data_path = "./combined_SOC_data_20250604.xlsx"  # 替换为您的实际文件路径
    
    # 训练集和测试集循环
    train_cycles = [2, 10, 16, 17, 43, 44]
    test_cycles = [79, 80]
    
    # 加载数据
    print("加载训练数据...")
    train_data = load_and_prepare_data(data_path, train_cycles)
    
    print("加载测试数据...")
    test_data = load_and_prepare_data(data_path, test_cycles)
    
    # 标准化数据
    train_data_scaled, test_data_scaled, scaler = standardize_data(train_data, test_data)
    
    # 创建数据集
    sequence_length = 20
    batch_size = 64
    
    print("创建训练数据集...")
    train_dataset = CycleDataset(train_data_scaled, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("创建测试数据集...")
    test_dataset = CycleDataset(test_data_scaled, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_dim = 3  # Current_A, Voltage_V, Cycle_Index
    output_dim = 1  # True_SOC
    model = iTransformer(input_dim, output_dim)
    # 在main函数中创建模型时增加复杂度
    # model = iTransformer(
    #     input_dim=3, 
    #     output_dim=1,
    #     d_model=128,  
    #     nhead=8,      
    #     num_layers=4,
    #     dropout=0.3
    # )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")
    
    epochs = 100
    patience = 10
    lr = 0.001
    
    # 训练模型
    print("开始训练模型...")
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, epochs, lr, patience
    )
    
    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('MSE损失')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/训练损失曲线.png', dpi=300)
    plt.close()
    
    # 对训练集进行预测和评估
    print("对训练集进行预测...")
    for cycle in train_cycles:
        if cycle in train_data_scaled:
            # 为每个循环创建单独的数据加载器
            cycle_data = {cycle: train_data_scaled[cycle]}
            cycle_dataset = CycleDataset(cycle_data, sequence_length)
            cycle_loader = DataLoader(cycle_dataset, batch_size=batch_size, shuffle=False)
            
            preds, targets = predict(trained_model, cycle_loader)
            
            # 计算评估指标
            mae = np.mean(np.abs(preds - targets))
            max_error = np.max(np.abs(preds - targets))
            print(f"Cycle {cycle} 训练集 MAE: {mae:.6f}, 最大误差: {max_error:.6f}")
            
            # 可视化并保存结果
            plot_results(preds, targets, cycle, 'train')
            save_predictions(preds, targets, cycle, 'train')
    
    # 对测试集进行预测和评估
    print("对测试集进行预测...")
    for cycle in test_cycles:
        if cycle in test_data_scaled:
            # 为每个循环创建单独的数据加载器
            cycle_data = {cycle: test_data_scaled[cycle]}
            cycle_dataset = CycleDataset(cycle_data, sequence_length)
            cycle_loader = DataLoader(cycle_dataset, batch_size=batch_size, shuffle=False)
            
            preds, targets = predict(trained_model, cycle_loader)
            
            # 计算评估指标
            mae = np.mean(np.abs(preds - targets))
            max_error = np.max(np.abs(preds - targets))
            print(f"Cycle {cycle} 测试集 MAE: {mae:.6f}, 最大误差: {max_error:.6f}")
            
            # 可视化并保存结果
            plot_results(preds, targets, cycle, 'test')
            save_predictions(preds, targets, cycle, 'test')
    
    # 保存模型
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler': scaler,
        'sequence_length': sequence_length
    }, 'results/soc_iTransformer_model.pth')
    
    print("模型已保存为 'results/soc_iTransformer_model.pth'")

if __name__ == "__main__":
    main()

# start
# Cycle 2 训练集 MAE: 0.009345, 最大误差: 0.021280
# Cycle 16 训练集 MAE: 0.007386, 最大误差: 0.018452
# Cycle 17 训练集 MAE: 0.008830, 最大误差: 0.024489
# Cycle 43 训练集 MAE: 0.015004, 最大误差: 0.027257
# Cycle 44 训练集 MAE: 0.011518, 最大误差: 0.024704
# 对测试集进行预测...
# Cycle 79 测试集 MAE: 0.034725, 最大误差: 0.073196
# Cycle 80 测试集 MAE: 0.040337, 最大误差: 0.101231

# 模型架构调整后
# Cycle 2 训练集 MAE: 0.012186, 最大误差: 0.041780
# Cycle 16 训练集 MAE: 0.016596, 最大误差: 0.023785
# Cycle 17 训练集 MAE: 0.008714, 最大误差: 0.020407
# Cycle 43 训练集 MAE: 0.026041, 最大误差: 0.052018
# Cycle 44 训练集 MAE: 0.009260, 最大误差: 0.027982
# 对测试集进行预测...
# Cycle 79 测试集 MAE: 0.026512, 最大误差: 0.069478
# Cycle 80 测试集 MAE: 0.020515, 最大误差: 0.059547
