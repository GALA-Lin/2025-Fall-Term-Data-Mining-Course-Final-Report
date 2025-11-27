import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 读取数据
df = pd.read_csv('penguins_size.csv')

# 2. 数据清洗
# 去除缺失值
df = df.dropna()
# 去除性别为 '.' 的脏数据
df = df[df['sex'] != '.']

# 3. 数值离散化
# 将数值特征分为 'Small', 'Medium', 'Large' 三个等级
labels = ['Low', 'Medium', 'High']
df['culmen_length_bin'] = pd.qcut(df['culmen_length_mm'], q=3, labels=labels)
df['culmen_depth_bin'] = pd.qcut(df['culmen_depth_mm'], q=3, labels=labels)
df['flipper_length_bin'] = pd.qcut(df['flipper_length_mm'], q=3, labels=labels)
df['body_mass_bin'] = pd.qcut(df['body_mass_g'], q=3, labels=labels)

# 保留离散化后的列和原始的分类列
cols_to_keep = ['species', 'island', 'sex',
                'culmen_length_bin', 'culmen_depth_bin',
                'flipper_length_bin', 'body_mass_bin']
df_binned = df[cols_to_keep]

# 4. One-Hot 编码
df_encoded = pd.get_dummies(df_binned, prefix_sep='=')

# 5. 运行 Apriori 算法
# min_support 设置为 0.1 (10%) 以捕获更多潜在模式，后面再过滤
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# 6. 生成关联规则
# metric 使用 'confidence' (置信度)，阈值设为 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 7. 筛选我们感兴趣的规则
# 增加一个 'lift' (提升度) 列
rules = rules.sort_values(by='confidence', ascending=False)

def filter_rules(rules, antecedent_keyword, consequent_keyword):
    filtered = rules[
        rules['antecedents'].apply(lambda x: any(antecedent_keyword in item for item in x)) &
        rules['consequents'].apply(lambda x: any(consequent_keyword in item for item in x))
    ]
    return filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("--- 数据集概况 ---")
print(f"清洗后剩余样本数: {len(df)}")