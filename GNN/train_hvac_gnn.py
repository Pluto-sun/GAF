import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from models.GNN import GNN
from data_provider.data_loader.gnn_hvac_loader import HVACGraphDataset

# 设置seaborn样式
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

def train_model(model, train_loader, test_loader, num_epochs=100, lr=0.001, device='cpu'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    best_test_acc = 0
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 25
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in train_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # 测试阶段
        test_acc = evaluate_model(model, test_loader, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        scheduler.step(train_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # 保存最佳模型和早停
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc
    }

def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    return correct / total

def detailed_evaluation(model, test_loader, device='cpu', num_classes=5):
    """详细评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 生成分类报告
    class_names = [f'异常类型_{i+1}' for i in range(num_classes)]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, np.array(all_probs)

def plot_results(history, save_path='training_results.png'):
    """绘制训练结果"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    sns.lineplot(data=pd.DataFrame({
        'Epoch': range(len(history['train_losses'])),
        'Loss': history['train_losses']
    }), x='Epoch', y='Loss', label='Train Loss')
    plt.title('Training Loss', pad=20)
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    df_acc = pd.DataFrame({
        'Epoch': list(range(len(history['train_accs']))) * 2,
        'Accuracy': history['train_accs'] + history['test_accs'],
        'Type': ['Train'] * len(history['train_accs']) + ['Test'] * len(history['test_accs'])
    })
    sns.lineplot(data=df_acc, x='Epoch', y='Accuracy', hue='Type')
    plt.title('Accuracy', pad=20)
    plt.grid(True)
    
    # 性能总结
    plt.subplot(1, 3, 3)
    final_train_acc = history['train_accs'][-1]
    best_test_acc = history['best_test_acc']
    
    performance_text = f"""
    Performance Summary:
    
    Final Train Acc: {final_train_acc:.4f}
    Best Test Acc: {best_test_acc:.4f}
    
    Total Epochs: {len(history['train_losses'])}
    
    Model: Graph Neural Network
    Task: HVAC Anomaly Detection
    """
    
    plt.text(0.1, 0.5, performance_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center')
    plt.title('Training Summary', pad=20)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, num_classes, save_path='confusion_matrix.png'):
    """绘制混淆矩阵（使用seaborn实现）"""
    plt.figure(figsize=(10, 8))
    class_names = [f'异常类型_{i+1}' for i in range(num_classes)]
    
    # 使用seaborn的heatmap绘制混淆矩阵
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - HVAC Anomaly Detection', pad=20)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='HVAC异常检测GNN训练 - 基于变量关联')
    parser.add_argument('--data_dir', type=str, required=True, help='数据文件夹路径')
    parser.add_argument('--correlation_threshold', type=float, default=0.3, help='相关性阈值')
    parser.add_argument('--sample_size', type=int, default=1000, help='每个异常类型的采样数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--use_attention', action='store_true', help='使用图注意力网络')
    parser.add_argument('--save_model', type=str, default='hvac_gnn_model.pth', help='模型保存路径')
    parser.add_argument('--results_dir', type=str, default='results', help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 1. 加载和处理数据
    print("=" * 60)
    print("步骤1: 数据加载和预处理")
    print("=" * 60)
    
    dataset = HVACGraphDataset(
        args.data_dir, 
        correlation_threshold=args.correlation_threshold,
        sample_size=args.sample_size
    )
    dataset.load_and_process_data()
    
    # 可视化图结构
    dataset.visualize_graph_structure(save_path=os.path.join(args.results_dir, 'graph_structure.png'))
    
    train_loader, test_loader = dataset.get_data_loaders(
        test_size=args.test_size, 
        batch_size=args.batch_size
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 2. 初始化模型
    print("=" * 60)
    print("步骤2: 模型初始化")
    print("=" * 60)
    
    model = HVACGraphNet(
        num_node_features=1,  # 每个节点只有一个特征值
        hidden_dim=args.hidden_dim,
        num_classes=dataset.num_classes,
        dropout=0.5,
        use_attention=args.use_attention
    )
    
    print(f"模型类型: {'Graph Attention Network' if args.use_attention else 'Graph Convolutional Network'}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 3. 训练模型
    print("=" * 60)
    print("步骤3: 模型训练")
    print("=" * 60)
    
    model, history = train_model(
        model, train_loader, test_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device
    )
    
    # 4. 详细评估
    print("=" * 60)
    print("步骤4: 模型评估")
    print("=" * 60)
    
    accuracy, report, cm, probs = detailed_evaluation(model, test_loader, device, num_classes=dataset.num_classes)
    
    print(f"\n最终测试准确率: {accuracy:.4f}")
    print(f"最佳测试准确率: {history['best_test_acc']:.4f}")
    print("\n分类报告:")
    print(report)
    
    # 5. 保存结果
    print("=" * 60)
    print("步骤5: 保存结果")
    print("=" * 60)
    
    # 保存模型
    model_save_path = os.path.join(args.results_dir, args.save_model)
    torch.save({
        'model_state_dict': model.state_dict(),
        'correlation_matrix': dataset.correlation_matrix,
        'adjacency_matrix': dataset.global_adj_matrix,
        'feature_names': dataset.feature_names,
        'scaler': dataset.scaler,
        'args': args
    }, model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 绘制结果
    plot_results(history, save_path=os.path.join(args.results_dir, 'training_results.png'))
    plot_confusion_matrix(cm, dataset.num_classes, save_path=os.path.join(args.results_dir, 'confusion_matrix.png'))
    
    # 保存训练历史
    history_save_path = os.path.join(args.results_dir, 'training_history.pkl')
    import pickle
    with open(history_save_path, 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"方法: 基于变量关联的图神经网络")
    print(f"图结构: 静态图（基于特征相关性）")
    print(f"最终准确率: {accuracy:.4f}")
    print(f"相比随机森林的改进: 考虑了变量间的图结构关系")
    print(f"\n所有结果已保存到目录: {args.results_dir}")

if __name__ == '__main__':
    main()