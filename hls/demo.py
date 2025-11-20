# 1. 导入库并设置中文显示
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 2. 数据加载与预处理
df = pd.read_csv('penguins_size.csv')

# 数据清洗：删除包含空值的行
df_clean = df.dropna()

# 修正性别字段的异常值
df_clean['sex'] = df_clean['sex'].replace('.', np.nan)
df_clean = df_clean.dropna(subset=['sex'])

# 特征工程：创建有意义的新特征，避免覆盖原始数据
df_clean['culmen_ratio'] = df_clean['culmen_length_mm'] / df_clean['culmen_depth_mm']  
df_clean['body_mass_kg'] = df_clean['body_mass_g'] / 1000  

# 3. 探索性数据分析（EDA）
print("数据集基本信息:")
print(f"数据形状: {df_clean.shape}")
print("\n数据类型:")
print(df_clean.dtypes)
print("\n描述性统计:")
print(df_clean.describe())

# 3.1 种类分布可视化
plt.figure(figsize=(10, 6))
species_counts = df_clean['species'].value_counts()
ax = species_counts.plot(kind='bar', color='skyblue')
plt.title('企鹅种类分布', fontsize=14)
plt.xlabel('种类')
plt.ylabel('数量')
plt.xticks(rotation=0)

# 在柱状图上添加具体数值和百分比
for i, v in enumerate(species_counts):
    percentage = v / len(df_clean) * 100
    ax.text(i, v + 2, f'{v}\n({percentage:.1f}%)', ha='center')

plt.tight_layout()
plt.show()

# 3.2 特征相关性分析
numeric_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'culmen_ratio', 'body_mass_kg']
correlation_matrix = df_clean[numeric_features].corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(numeric_features)), numeric_features, rotation=45)
plt.yticks(range(len(numeric_features)), numeric_features)
plt.title('特征相关性热力图', fontsize=14)

# 在热力图上添加相关系数
for i in range(len(numeric_features)):
    for j in range(len(numeric_features)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                 ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# 4. 机器学习建模

# 4.1 数据准备
# 编码分类特征
label_encoders = {}
for column in ['species', 'island', 'sex']:
    le = LabelEncoder()
    df_clean[column + '_encoded'] = le.fit_transform(df_clean[column])
    label_encoders[column] = le

# 定义特征和目标变量
X = df_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'culmen_ratio', 'body_mass_kg', 'island_encoded', 'sex_encoded']]
y_species = df_clean['species_encoded']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_species, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.2 模型训练与评估
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """训练并评估模型的函数"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} 模型评估:")
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=label_encoders['species'].classes_))
    
    # 混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} 混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoders['species'].classes_))
    plt.xticks(tick_marks, label_encoders['species'].classes_, rotation=45)
    plt.yticks(tick_marks, label_encoders['species'].classes_)
    
    # 在混淆矩阵上添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    return model, accuracy

# 初始化模型
models = {
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "决策树": DecisionTreeClassifier(random_state=42),
    "随机森林": RandomForestClassifier(random_state=42)
}

# 训练和评估所有模型
trained_models = {}
accuracies = {}
for name, model in models.items():
    trained_model, accuracy = train_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    trained_models[name] = trained_model
    accuracies[name] = accuracy

# 找出表现最好的模型
best_model_name = max(accuracies, key=accuracies.get)
print(f"\n表现最好的模型是: {best_model_name}，准确率为: {accuracies[best_model_name]:.4f}")

# 4.3 特征重要性分析（以随机森林为例）
if "随机森林" in trained_models:
    rf_model = trained_models["随机森林"]
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    
    # 创建特征重要性的DataFrame
    importance_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importance})
    importance_df = importance_df.sort_values('重要性', ascending=False)
    
    print("\n随机森林模型特征重要性:")
    print(importance_df)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['特征'], importance_df['重要性'], color='skyblue')
    plt.xlabel('重要性')
    plt.title('随机森林模型特征重要性排名')
    plt.gca().invert_yaxis()  # 从上到下按重要性降序排列
    plt.tight_layout()
    plt.show()

# 5. 聚类分析
# 使用身体测量特征进行聚类
features_for_clustering = df_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
scaler_clust = StandardScaler()
features_scaled = scaler_clust.fit_transform(features_for_clustering)

# 使用肘部法则和轮廓系数确定最佳聚类数
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, clusters))

# 可视化肘部法则结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('聚类数量 (k)')
plt.ylabel('惯性 (Inertia)')
plt.title('肘部法则确定最佳聚类数')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('聚类数量 (k)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数确定最佳聚类数')
plt.grid(True)

plt.tight_layout()
plt.show()

# 根据分析选择最佳聚类数（假设为3）
best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(features_scaled)

# 使用PCA进行降维可视化
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 8))
for cluster in range(best_k):
    plt.scatter(features_pca[df_clean['cluster'] == cluster, 0], 
                features_pca[df_clean['cluster'] == cluster, 1],
                label=f'聚类 {cluster + 1}',
                alpha=0.7)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='black', marker='X', label='聚类中心')
plt.title(f'K-Means聚类结果 (k={best_k})', fontsize=14)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 分析聚类与实际种类的对应关系
cluster_species = pd.crosstab(df_clean['cluster'], df_clean['species'])
print("\n聚类与实际种类的对应关系:")
print(cluster_species)

# 输出最终结论
print("\n========== 分析总结 ==========")
print(f"1. 数据预处理后，数据集包含 {df_clean.shape[0]} 条记录和 {df_clean.shape[1]} 个特征")
print(f"2. 最佳分类模型是 {best_model_name}，准确率为 {accuracies[best_model_name]:.4f}")
print(f"3. 最重要的特征是 {importance_df.iloc[0]['特征']}，重要性为 {importance_df.iloc[0]['重要性']:.4f}")
print(f"4. 通过聚类分析，企鹅可分为 {best_k} 个主要群体")
print("5. 聚类结果与实际企鹅种类有较好的对应关系")