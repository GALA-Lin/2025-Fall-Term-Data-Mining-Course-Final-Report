# ==============================================================
# 修正后：分类模型构建（统一使用body_mass_kg字段）
# ==============================================================

# 1. 导入库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 加载并准备数据（确保使用body_mass_kg）
df = pd.read_csv('penguins_eda_processed.csv')

# 特征集：明确使用body_mass_kg
feature_cols = [
    'culmen_length_mm', 'culmen_depth_mm', 
    'flipper_length_mm', 'body_mass_kg',  # 修正为kg单位
    'culmen_ratio'
]
X = df[feature_cols]

# 目标变量
y_species = df['species']  # 种类预测（多分类）
y_sex = df['sex']          # 性别预测（二分类）

# 3. 划分训练集与测试集
# 种类预测
X_train_species, X_test_species, y_train_species, y_test_species = train_test_split(
    X, y_species, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_species
)

# 性别预测
X_train_sex, X_test_sex, y_train_sex, y_test_sex = train_test_split(
    X, y_sex, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_sex
)

# 4. 训练3种模型（逻辑回归、决策树、随机森林）
# 初始化模型
model_logreg = LogisticRegression(max_iter=1000)
model_dt = DecisionTreeClassifier(random_state=42)
model_rf = RandomForestClassifier(random_state=42)

# 种类预测模型训练
model_logreg_species = model_logreg.fit(X_train_species, y_train_species)
model_dt_species = model_dt.fit(X_train_species, y_train_species)
model_rf_species = model_rf.fit(X_train_species, y_train_species)

# 性别预测模型训练
model_logreg_sex = model_logreg.fit(X_train_sex, y_train_sex)
model_dt_sex = model_dt.fit(X_train_sex, y_train_sex)
model_rf_sex = model_rf.fit(X_train_sex, y_train_sex)

# 5. 10折交叉验证评估稳定性
def cross_val_evaluate(model, X_train, y_train, task_name):
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    print(f"\n【{task_name}】10折交叉验证结果：")
    print(f"平均准确率：{cv_scores.mean():.4f}")
    print(f"准确率标准差：{cv_scores.std():.4f}")
    return cv_scores.mean(), cv_scores.std()

# 种类预测交叉验证
logreg_species_mean, logreg_species_std = cross_val_evaluate(model_logreg_species, X_train_species, y_train_species, "逻辑回归（种类预测）")
dt_species_mean, dt_species_std = cross_val_evaluate(model_dt_species, X_train_species, y_train_species, "决策树（种类预测）")
rf_species_mean, rf_species_std = cross_val_evaluate(model_rf_species, X_train_species, y_train_species, "随机森林（种类预测）")

# 性别预测交叉验证
logreg_sex_mean, logreg_sex_std = cross_val_evaluate(model_logreg_sex, X_train_sex, y_train_sex, "逻辑回归（性别预测）")
dt_sex_mean, dt_sex_std = cross_val_evaluate(model_dt_sex, X_train_sex, y_train_sex, "决策树（性别预测）")
rf_sex_mean, rf_sex_std = cross_val_evaluate(model_rf_sex, X_train_sex, y_train_sex, "随机森林（性别预测）")

# 6. 测试集最终评估
def final_evaluation(model, X_test, y_test, task_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n【{task_name}】测试集准确率：{accuracy:.4f}")
    print("分类报告：")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"【{task_name}】混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()
    return accuracy

# 种类预测最终评估
logreg_species_acc = final_evaluation(model_logreg_species, X_test_species, y_test_species, "逻辑回归（种类预测）")
dt_species_acc = final_evaluation(model_dt_species, X_test_species, y_test_species, "决策树（种类预测）")
rf_species_acc = final_evaluation(model_rf_species, X_test_species, y_test_species, "随机森林（种类预测）")

# 性别预测最终评估
logreg_sex_acc = final_evaluation(model_logreg_sex, X_test_sex, y_test_sex, "逻辑回归（性别预测）")
dt_sex_acc = final_evaluation(model_dt_sex, X_test_sex, y_test_sex, "决策树（性别预测）")
rf_sex_acc = final_evaluation(model_rf_sex, X_test_sex, y_test_sex, "随机森林（性别预测）")

# 7. 结果汇总
results_df = pd.DataFrame({
    "任务": ["种类预测", "种类预测", "种类预测", "性别预测", "性别预测", "性别预测"],
    "模型": ["逻辑回归", "决策树", "随机森林", "逻辑回归", "决策树", "随机森林"],
    "交叉验证准确率(均值)": [
        logreg_species_mean, dt_species_mean, rf_species_mean,
        logreg_sex_mean, dt_sex_mean, rf_sex_mean
    ],
    "测试集准确率": [
        logreg_species_acc, dt_species_acc, rf_species_acc,
        logreg_sex_acc, dt_sex_acc, rf_sex_acc
    ]
})

print("模型结果汇总：")
print(results_df.round(4))

# 8. 特征重要性分析（以随机森林为例）
rf_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': model_rf_species.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n随机森林特征重要性（种类预测）：")
print(rf_importance)