import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 140
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def get_prediction_rules():
    try:
        df = pd.read_csv('penguins_size.csv').dropna()
    except FileNotFoundError:
        return pd.DataFrame()
    df = df[df['sex'] != '.']

    # 离散化
    df_clean = pd.DataFrame()
    df_clean['物种'] = df['species']
    df_clean['岛屿'] = df['island']
    labels = ['短', '中', '长']
    labels_mass = ['轻', '中', '重']
    labels_depth = ['浅', '中', '深']

    df_clean['嘴长'] = pd.qcut(df['culmen_length_mm'], q=3, labels=labels)
    df_clean['嘴深'] = pd.qcut(df['culmen_depth_mm'], q=3, labels=labels_depth)
    df_clean['鳍长'] = pd.qcut(df['flipper_length_mm'], q=3, labels=labels)
    df_clean['体重'] = pd.qcut(df['body_mass_g'], q=3, labels=labels_mass)

    # 挖掘
    df_encoded = pd.get_dummies(df_clean).astype(bool)
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    if frequent_itemsets.empty: return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    rules['ant_str'] = rules['antecedents'].apply(lambda x: " + ".join(list(x)))
    rules['con_str'] = rules['consequents'].apply(lambda x: list(x)[0])

    target_species = ['物种_Adelie', '物种_Chinstrap', '物种_Gentoo']
    filtered = rules[rules['consequents'].apply(lambda x: len(x) == 1 and list(x)[0] in target_species)]

    return filtered.sort_values(by='lift', ascending=False)


def plot_beautiful_network(rules_df):
    if rules_df.empty: return

    # 取每个物种最强的 5 条规则
    top_rules = rules_df.groupby('con_str').head(5).reset_index(drop=True)

    G = nx.DiGraph()
    for _, row in top_rules.iterrows():
        ant_label = row['ant_str'].replace(" + ", "\n")
        con_label = row['con_str'].replace("物种_", "")
        G.add_edge(ant_label, con_label, weight=row['confidence'])

    # --- 布局算法 ---
    subgraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    pos = {}
    cluster_centers = [(-1.5, -0.6), (1.5, -0.6), (0, 1.0)]

    for i, sub_g in enumerate(subgraphs):
        center = cluster_centers[i % len(cluster_centers)]
        sub_pos = nx.spring_layout(sub_g, k=2.0, seed=42, iterations=100)
        for node, coords in sub_pos.items():
            pos[node] = (coords[0] + center[0], coords[1] + center[1])

    plt.figure(figsize=(12, 10))

    # 1. 定义配色
    color_species = '#E07A5F'
    color_island = '#81B29A'
    color_feature = '#F2CC8F'

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        if node in ['Adelie', 'Gentoo', 'Chinstrap']:
            node_colors.append(color_species)
            node_sizes.append(4000)
        elif '岛屿' in node:
            node_colors.append(color_island)
            node_sizes.append(3600)
        else:
            node_colors.append(color_feature)
            node_sizes.append(3600)

    # 2. 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] * 2 for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        node_size=node_sizes,
        arrowstyle='-|>',
        arrowsize=18,
        width=weights,
        edge_color='#999999',
        connectionstyle="arc3,rad=0.15",
        alpha=0.5,
        min_source_margin=15,
        min_target_margin=15
    )

    # 3. 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=2.5,
        alpha=1.0
    )

    # 4. 绘制文字
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_family='sans-serif',
        font_weight='bold',
        font_color='#333333'
    )

    # 5. 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_species, markersize=16, label='目标 (物种)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_island, markersize=14, label='依据 (岛屿)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_feature, markersize=14, label='依据 (形态)'),
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=False, labelspacing=1.5)

    plt.title("企鹅物种特征图谱", fontsize=20, color='#333333', pad=20, fontweight='bold')

    plt.text(0, -2.2, "线条粗细代表置信度 | 颜色代表数据类型",
             ha='center', color='#888888', fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rules = get_prediction_rules()
    if not rules.empty:
        plot_beautiful_network(rules)
    else:
        print("无规则")