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

# Set up Chinese font display (Note: This section remains for compatibility but comments are in English)
font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # Replace with actual path if needed
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# Create results directory
os.makedirs('results', exist_ok=True)

# 1. Define Transformer model architecture
class iTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(iTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.BatchNorm1d(d_model),  # Add BatchNorm
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model, output_dim)
        # )
    
    def forward(self, src):
        # Embedding layer
        src = self.embedding(src)
        
        # Positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src)
        
        # Take the output of the last time step in the sequence
        output = output[:, -1, :]
        
        # Fully connected layer
        output = self.fc(output)
        return output

# Positional encoding layer
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

# 2. Dataset class
class CycleDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        """
        Initialize dataset
        
        Args:
        data -- Dictionary containing all cycle data {cycle_index: (features, targets)}
        sequence_length -- Input sequence length
        """
        self.sequences = []
        self.targets = []
        
        for cycle_index, (features, target) in data.items():
            # Create sequences
            for i in range(len(features) - sequence_length):
                seq = features[i:i+sequence_length]
                label = target[i+sequence_length-1]  # Use true SOC at the last time step of the sequence
                
                self.sequences.append(seq)
                self.targets.append(label)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target

# 3. Data processing function
def load_and_prepare_data(file_path, selected_cycles=None):
    """
    Load and preprocess data
    
    Args:
    file_path -- Path to Excel file
    selected_cycles -- List of selected cycles (default None means load all cycles)
    
    Returns:
    cycle_data -- Dictionary {cycle_index: (features, targets)}
    """
    # Load data
    df = pd.read_excel(file_path)
    print(f"Data loading completed, total rows: {len(df)}")
    
    # Ensure correct column names
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
    
    # Rename columns
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Keep only necessary columns
    keep_columns = ['Cycle_Index', 'Current_A', 'Voltage_V', 'True_SOC']
    if 'Data_Point' in df.columns:
        keep_columns.append('Data_Point')
    else:
        # Add row index as fallback sorting key
        df['Data_Point'] = df.index
    
    # Drop irrelevant columns
    drop_columns = [col for col in df.columns if col not in keep_columns]
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    # Filter specified cycles
    if selected_cycles:
        df = df[df['Cycle_Index'].isin(selected_cycles)]
    
    # Group by cycle
    grouped = df.groupby('Cycle_Index')
    
    # Prepare data dictionary
    cycle_data = {}
    
    # Store data per cycle, ensuring order within each cycle
    for cycle_index, group in grouped:
        # If no 'Data_Point' column, use original index order
        group = group.sort_values('Data_Point')  # Ensure data point order

        # Extract features and targets
        features = group[['Current_A', 'Voltage_V', 'Cycle_Index']].values
        targets = group['True_SOC'].values
        
        cycle_data[cycle_index] = (features, targets)
    
    print(f"Data preprocessing completed, loaded data from {len(cycle_data)} cycles")
    return cycle_data

# 4. Data standardization
def standardize_data(train_data, test_data):
    """
    Standardize training and testing data
    
    Args:
    train_data -- Training data dictionary
    test_data -- Testing data dictionary
    
    Returns:
    train_data_scaled -- Standardized training data
    test_data_scaled -- Standardized testing data
    scaler -- Scaler object for inverse transformation
    """
    # Collect all features to fit the scaler
    all_features = []
    for features, _ in train_data.values():
        all_features.extend(features)
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Standardize training data
    train_data_scaled = {}
    for cycle, (features, targets) in train_data.items():
        features_scaled = scaler.transform(features)
        train_data_scaled[cycle] = (features_scaled, targets)
    
    # Standardize testing data
    test_data_scaled = {}
    for cycle, (features, targets) in test_data.items():
        features_scaled = scaler.transform(features)
        test_data_scaled[cycle] = (features_scaled, targets)
    
    return train_data_scaled, test_data_scaled, scaler

# 5. Model training function - Modified learning rate scheduler usage
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):
    """
    Train the model
    
    Args:
    model -- Model to be trained
    train_loader -- Training data loader
    val_loader -- Validation data loader
    epochs -- Number of training epochs
    lr -- Learning rate
    patience -- Patience for early stopping
    
    Returns:
    train_losses -- List of training losses
    val_losses -- List of validation losses
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # Add L2 regularization
    
    # Learning rate scheduler - Removed verbose argument
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=patience//2
    # )
    # Increase patience for learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=16
    )
    
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    lr_history = []  # Record learning rate changes
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training phase
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
        
        # Validation phase
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
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Check if learning rate changed
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            tqdm.write(f"Learning rate reduced to {new_lr:.7f} at epoch {epoch+1}")
        
        # Record current learning rate
        lr_history.append(new_lr)
        
        # Print progress
        tqdm.write(f'Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | LR: {new_lr:.6f}')
        
        # Early stopping check
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
    
    # Plot learning rate change curve
    plt.figure(figsize=(10, 4))
    plt.plot(lr_history, 'b-o')
    plt.title('Learning Rate Change During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/Learning_Rate_Curve.png', dpi=300)
    plt.close()
    
    return model, train_losses, val_losses

# 6. Prediction function
def predict(model, data_loader):
    """
    Make predictions using the model
    
    Args:
    model -- Trained model
    data_loader -- Data loader
    
    Returns:
    all_preds -- All predicted values
    all_targets -- All true values
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

# 7. Visualization function
def plot_results(preds, targets, cycle_index, dataset_type):
    """
    Visualize prediction results
    
    Args:
    preds -- Array of predicted values
    targets -- Array of true values
    cycle_index -- Cycle index
    dataset_type -- Dataset type ('train' or 'test')
    """
    plt.figure(figsize=(14, 8))
    
    # Comparison of true vs predicted SOC
    plt.subplot(2, 2, (1, 2))
    plt.plot(targets, 'b-', alpha=0.7, label='True SOC')
    plt.plot(preds, 'r--', alpha=0.9, label='Predicted SOC')
    plt.title(f'Cycle {cycle_index} {dataset_type} SOC Prediction Comparison')
    plt.xlabel('Data Point Index')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)
    
    # Prediction error distribution
    plt.subplot(2, 2, 3)
    errors = preds - targets
    sns.histplot(errors, kde=True)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.grid(True)
    
    # Error scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(targets, preds, alpha=0.3)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.title('True vs Predicted Values')
    plt.xlabel('True SOC')
    plt.ylabel('Predicted SOC')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/Cycle_{cycle_index}_{dataset_type}_Prediction_Results.png', dpi=300)
    plt.close()

# 8. Save prediction results
def save_predictions(preds, targets, cycle_index, dataset_type):
    """
    Save prediction results to CSV file
    
    Args:
    preds -- Array of predicted values
    targets -- Array of true values
    cycle_index -- Cycle index
    dataset_type -- Dataset type ('train' or 'test')
    """
    results = pd.DataFrame({
        'Data_Point_Index': np.arange(len(preds)),
        'Predicted_SOC': preds,
        'True_SOC': targets
    })
    results.to_csv(f'results/Cycle_{cycle_index}_{dataset_type}_Prediction_Results.csv', index=False)

# 9. Main function
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define dataset path
    data_path = "./combined_SOC_data_20250604.xlsx"  # Replace with your actual file path
    
    # Training and testing cycles
    train_cycles = [2, 10, 16, 17, 43, 44]
    test_cycles = [79, 80]
    
    # Load data
    print("Loading training data...")
    train_data = load_and_prepare_data(data_path, train_cycles)
    
    print("Loading testing data...")
    test_data = load_and_prepare_data(data_path, test_cycles)
    
    # Standardize data
    train_data_scaled, test_data_scaled, scaler = standardize_data(train_data, test_data)
    
    # Create datasets
    sequence_length = 20
    batch_size = 64
    
    print("Creating training dataset...")
    train_dataset = CycleDataset(train_data_scaled, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Creating testing dataset...")
    test_dataset = CycleDataset(test_data_scaled, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = 3  # Current_A, Voltage_V, Cycle_Index
    output_dim = 1  # True_SOC
    model = iTransformer(input_dim, output_dim)
    # In main function, increase model complexity when creating model
    # model = iTransformer(
    #     input_dim=3, 
    #     output_dim=1,
    #     d_model=128,  
    #     nhead=8,      
    #     num_layers=4,
    #     dropout=0.3
    # )
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    epochs = 100
    patience = 10
    lr = 0.001
    
    # Train model
    print("Starting model training...")
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, epochs, lr, patience
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/Training_Loss_Curve.png', dpi=300)
    plt.close()
    
    # Predict and evaluate on training set
    print("Making predictions on training set...")
    for cycle in train_cycles:
        if cycle in train_data_scaled:
            # Create separate data loader for each cycle
            cycle_data = {cycle: train_data_scaled[cycle]}
            cycle_dataset = CycleDataset(cycle_data, sequence_length)
            cycle_loader = DataLoader(cycle_dataset, batch_size=batch_size, shuffle=False)
            
            preds, targets = predict(trained_model, cycle_loader)
            
            # Compute evaluation metrics
            mae = np.mean(np.abs(preds - targets))
            max_error = np.max(np.abs(preds - targets))
            print(f"Cycle {cycle} Training MAE: {mae:.6f}, Max Error: {max_error:.6f}")
            
            # Visualize and save results
            plot_results(preds, targets, cycle, 'train')
            save_predictions(preds, targets, cycle, 'train')
    
    # Predict and evaluate on test set
    print("Making predictions on test set...")
    for cycle in test_cycles:
        if cycle in test_data_scaled:
            # Create separate data loader for each cycle
            cycle_data = {cycle: test_data_scaled[cycle]}
            cycle_dataset = CycleDataset(cycle_data, sequence_length)
            cycle_loader = DataLoader(cycle_dataset, batch_size=batch_size, shuffle=False)
            
            preds, targets = predict(trained_model, cycle_loader)
            
            # Compute evaluation metrics
            mae = np.mean(np.abs(preds - targets))
            max_error = np.max(np.abs(preds - targets))
            print(f"Cycle {cycle} Test MAE: {mae:.6f}, Max Error: {max_error:.6f}")
            
            # Visualize and save results
            plot_results(preds, targets, cycle, 'test')
            save_predictions(preds, targets, cycle, 'test')
    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler': scaler,
        'sequence_length': sequence_length
    }, 'results/soc_iTransformer_model.pth')
    
    print("Model saved as 'results/soc_iTransformer_model.pth'")

if __name__ == "__main__":
    main()