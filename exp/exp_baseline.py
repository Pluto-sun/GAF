import argparse
import numpy as np
from data_provider.data_loader import RawFeatureWindowLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline ML for Anomaly Diagnosis")
    parser.add_argument("--root_path", type=str, required=True, help="数据根目录")
    parser.add_argument("--win_size", type=int, required=True, help="窗口大小")
    parser.add_argument("--step", type=int, required=True, help="滑动步长")
    parser.add_argument("--num_class", type=int, required=True, help="类别数")
    parser.add_argument("--val_ratio", type=float, default=0.3, help="验证集比例")
    args = parser.parse_args()

    # 使用RawFeatureWindowLoader加载原始特征窗口
    loader = RawFeatureWindowLoader(args.root_path, args.win_size, args.step)
    (X_train, y_train), (X_val, y_val) = loader.get_train_val()

    # 特征降维：对每个窗口取均值，得到[N, D]特征
    X_train_flat = X_train
    X_val_flat = X_val

    print("训练集样本数:", X_train_flat.shape[0])
    print("验证集样本数:", X_val_flat.shape[0])
    print("特征维度:", X_train_flat.shape[1])
    print("类别分布:", np.unique(y_train, return_counts=True))

    # 1. 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train)
    y_pred_rf = rf.predict(X_val_flat)
    print("\n[随机森林] 准确率:", accuracy_score(y_val, y_pred_rf))
    print(classification_report(y_val, y_pred_rf))

    # 2. SVM
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train_flat, y_train)
    y_pred_svm = svm.predict(X_val_flat)
    print("\n[SVM] 准确率:", accuracy_score(y_val, y_pred_svm))
    print(classification_report(y_val, y_pred_svm))

    # 3. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_flat, y_train)
    y_pred_knn = knn.predict(X_val_flat)
    print("\n[KNN] 准确率:", accuracy_score(y_val, y_pred_knn))
    print(classification_report(y_val, y_pred_knn))

    # 4. 逻辑回归
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_flat, y_train)
    y_pred_lr = lr.predict(X_val_flat)
    print("\n[逻辑回归] 准确率:", accuracy_score(y_val, y_pred_lr))
    print(classification_report(y_val, y_pred_lr))
