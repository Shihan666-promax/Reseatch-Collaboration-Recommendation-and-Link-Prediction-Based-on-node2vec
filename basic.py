import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_data(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                u, v = map(int, line.strip().split())
                edges.append((u, v))
    return edges


# 加载数据并构建原始网络
edges = load_data("D:/network/ca-CondMat/CA-CondMat.txt")
G_original = nx.Graph()
G_original.add_edges_from(edges)
print(f"原始网络节点数：{G_original.number_of_nodes()}")
print(f"原始网络边数：{G_original.number_of_edges()}")

# 原始网络结构可视化
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_original, seed=42, k=0.30)
nx.draw_networkx_nodes(G_original, pos, node_size=20, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G_original, pos, alpha=0.2, width=0.5)
plt.title('原始网络')
plt.axis('off')
plt.tight_layout()
plt.savefig('D:/network/原始网络.png', dpi=300)
plt.show()

def calculate_metrics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    avg_degree = np.mean(degree_values) if degree_values else 0
    degree_std = np.std(degree_values) if degree_values else 0

    # 连通性分析
    connected_components = list(nx.connected_components(G))
    num_connected_components = len(connected_components)
    largest_component = max(connected_components, key=len) if connected_components else []
    G_largest = G.subgraph(largest_component)

    # 路径与距离指标
    try:
        avg_shortest_path = nx.average_shortest_path_length(G_largest) if len(largest_component) > 1 else 0
    except:
        avg_shortest_path = 0
    try:
        diameter = nx.diameter(G_largest) if len(largest_component) > 1 else 0
    except:
        diameter = 0

    # 聚类系数
    node_clustering = nx.clustering(G)
    avg_clustering = np.mean(list(node_clustering.values())) if node_clustering else 0
    transitivity = nx.transitivity(G)

    metrics = {
        "节点数": num_nodes,
        "边数": num_edges,
        "网络密度": density,
        "平均度": avg_degree,
        "度的标准差": degree_std,
        "连通分量数": num_connected_components,
        "平均最短路径长度": avg_shortest_path,
        "网络直径": diameter,
        "平均聚类系数": avg_clustering,
        "传递性": transitivity,
    }

    return metrics, degrees, node_clustering


# 计算原始网络指标
original_metrics, original_degrees, original_node_clustering = calculate_metrics(G_original)
print("\n原始网络统计指标：")
for k, v in original_metrics.items():
    if v is not None:
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        print(f"{k}: 未计算（节点数量过多）")

# 原始网络度分布可视化
def visualize_original_degree_distribution(degrees, title_suffix, save_suffix):
    degree_values = list(degrees.values())
    plt.figure(figsize=(10, 6))

    # 直方图
    plt.hist(degree_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('节点度')
    plt.ylabel('节点数量')
    plt.title(f'原始网络度分布')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'D:/network/原始网络度分布.png', dpi=300)
    plt.show()


# 可视化原始网络度分布
visualize_original_degree_distribution(original_degrees)


# 原始网络聚类系数分布可视化
def visualize_original_clustering_distribution(node_clustering):
    clustering_values = list(node_clustering.values())
    clustering_values = [c for c in clustering_values if c > 0]

    plt.figure(figsize=(10, 6))
    plt.hist(clustering_values, bins=50, alpha=0.7, color='yellow', edgecolor='black')
    plt.xlabel('节点聚类系数')
    plt.ylabel('节点数量')
    plt.title(f'原始网络聚类系数分布')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'D:/network/原始网络聚类系数分布.png', dpi=300)
    plt.show()

# 可视化原始网络系数分布
visualize_original_clustering_distribution(original_node_clustering)


# 获取原始网络最大连通分量前500个节点
def get_largest_component_top_nodes(G, top_k=500):
    # 找到最大连通分量
    largest_component = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_component)

    # 按节点度排序，取前500个
    nodes_sorted_by_degree = sorted(G_largest.nodes(), key=lambda x: G_largest.degree(x), reverse=True)
    top_nodes = nodes_sorted_by_degree[:top_k]
    G_top = G_largest.subgraph(top_nodes)

    print(f"\n最大连通分量节点数：{len(largest_component)}")
    print(f"选取最大连通分量前{len(top_nodes)}个节点构建子网络")
    print(f"子网络节点数：{G_top.number_of_nodes()}")
    print(f"子网络边数：{G_top.number_of_edges()}")
    return G_top


# 获取原始网络最大连通分量前500个节点子网络
G_original_top500 = get_largest_component_top_nodes(G_original, top_k=500)



# 生成对比网络
def generate_comparison_networks(n_nodes_full, n_edges_full, n_nodes_top500, n_edges_top500):
    print("\n生成全节点随机网络...")
    # 全节点随机网络
    p_full = 2 * n_edges_full / (n_nodes_full * (n_nodes_full - 1))
    G_random_full = nx.erdos_renyi_graph(n=n_nodes_full, p=p_full, seed=42)
    # 可视化全节点随机网络
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_random_full, seed=42, k=0.30)
    nx.draw_networkx_nodes(G_random_full, pos, node_size=20, node_color='orange', alpha=0.8)
    nx.draw_networkx_edges(G_random_full, pos, alpha=0.2, width=0.5)
    plt.title('随机网络')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('D:/network/随机网络.png', dpi=300)
    plt.show()

    print("\n生成全节点无标度网络...")
    # 全节点无标度网络
    m_full = int(n_edges_full / n_nodes_full)
    m_full = max(1, m_full)
    G_scale_free_full = nx.barabasi_albert_graph(n=n_nodes_full, m=m_full, seed=42)
    # 可视化全节点无标度网络
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_scale_free_full, seed=42, k=0.30)
    nx.draw_networkx_nodes(G_scale_free_full, pos, node_size=20, node_color='yellow', alpha=0.8)
    nx.draw_networkx_edges(G_scale_free_full, pos, alpha=0.2, width=0.5)
    plt.title('无标度网络')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('D:/network/无标度网络.png', dpi=300)
    plt.show()

    print("\n生成最大连通分量前500节点随机网络...")
    p_top500 = 2 * n_edges_top500 / (n_nodes_top500 * (n_nodes_top500 - 1))
    G_random_top500 = nx.erdos_renyi_graph(n=n_nodes_top500, p=p_top500, seed=42)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_random_top500, seed=42, k=0.4)
    nx.draw_networkx_nodes(G_random_top500, pos, node_size=30, node_color='coral', alpha=0.9)
    nx.draw_networkx_edges(G_random_top500, pos, alpha=0.3, width=0.6)
    plt.title('随机网络（最大连通分量前500节点）')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('D:/network/随机网络_前500节点.png', dpi=300)
    plt.show()

    print("\n生成最大连通分量前500节点无标度网络...")
    m_top500 = int(n_edges_top500 / n_nodes_top500)
    m_top500 = max(1, m_top500)
    G_scale_free_top500 = nx.barabasi_albert_graph(n=n_nodes_top500, m=m_top500, seed=42)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_scale_free_top500, seed=42, k=0.4)
    nx.draw_networkx_nodes(G_scale_free_top500, pos, node_size=30, node_color='gold', alpha=0.9)
    nx.draw_networkx_edges(G_scale_free_top500, pos, alpha=0.3, width=0.6)
    plt.title('无标度网络（最大连通分量前500节点）')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('D:/network/无标度网络_前500节点.png', dpi=300)
    plt.show()

    return G_random_full, G_scale_free_full, G_random_top500, G_scale_free_top500


# 生成对比网络
n_nodes_full = G_original.number_of_nodes()
n_edges_full = G_original.number_of_edges()
n_nodes_top500 = G_original_top500.number_of_nodes()
n_edges_top500 = G_original_top500.number_of_edges()

print(f"\n生成全节点对比网络（节点数: {n_nodes_full}, 目标边数: {n_edges_full}）")
print(f"\n生成前500节点对比网络（节点数: {n_nodes_top500}, 目标边数: {n_edges_top500}）")

G_random_full, G_scale_free_full, G_random_top500, G_scale_free_top500 = generate_comparison_networks(
    n_nodes_full, n_edges_full, n_nodes_top500, n_edges_top500
)

print(f"\n全节点随机网络实际边数：{G_random_full.number_of_edges()}")
print(f"全节点无标度网络实际边数：{G_scale_free_full.number_of_edges()}")
print(f"前500节点随机网络实际边数：{G_random_top500.number_of_edges()}")
print(f"前500节点无标度网络实际边数：{G_scale_free_top500.number_of_edges()}")


def compare_key_network_metrics(original_metrics, random_metrics, scale_free_metrics):
    metrics_to_compare = [
        "平均最短路径长度",
        "平均聚类系数"
    ]

    # 准备数据
    data = {
        "原始网络": [original_metrics[m] for m in metrics_to_compare],
        "随机网络": [random_metrics[m] for m in metrics_to_compare],
        "无标度网络": [scale_free_metrics[m] for m in metrics_to_compare]
    }

    # 绘制对比图
    x = np.arange(len(metrics_to_compare))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, data["原始网络"], width, label='原始网络', color='skyblue')
    plt.bar(x, data["随机网络"], width, label='随机网络', color='orange')
    plt.bar(x + width, data["无标度网络"], width, label='无标度网络', color='red')

    plt.xlabel('网络指标')
    plt.ylabel('指标值')
    plt.title('三种网络核心指标对比')
    plt.xticks(x, metrics_to_compare)
    plt.legend()
    plt.tight_layout()
    plt.savefig('D:/network/三种网络核心指标对比.png', dpi=300)
    plt.show()

    print("\n三种网络核心指标数值对比：")
    comparison_df = pd.DataFrame(data, index=metrics_to_compare)
    print(comparison_df.round(4))


# 计算全节点对比网络的指标
random_full_metrics, random_full_degrees, _ = calculate_metrics(G_random_full)
scale_free_full_metrics, scale_free_full_degrees, _ = calculate_metrics(G_scale_free_full)

# 仅对比全节点规模的核心指标
compare_key_network_metrics(original_metrics, random_full_metrics, scale_free_full_metrics)


# 对比三种网络的度分布
def compare_degree_distributions(original_deg, random_deg, scale_free_deg):
    # 提取度值
    original_vals = list(original_deg.values())
    random_vals = list(random_deg.values())
    scale_free_vals = list(scale_free_deg.values())

    plt.figure(figsize=(15, 5))

    # 原始网络
    plt.subplot(1, 3, 1)
    plt.hist(original_vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('原始网络度分布')
    plt.xlabel('节点度')
    plt.ylabel('节点数量')
    plt.grid(alpha=0.3)

    # 随机网络
    plt.subplot(1, 3, 2)
    plt.hist(random_vals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('随机网络度分布')
    plt.xlabel('节点度')
    plt.ylabel('节点数量')
    plt.grid(alpha=0.3)

    # 无标度网络
    plt.subplot(1, 3, 3)
    plt.hist(scale_free_vals, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.title('无标度网络度分布')
    plt.xlabel('节点度')
    plt.ylabel('节点数量')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/network/度分布.png', dpi=300)
    plt.show()

    # 对数坐标对比
    plt.figure(figsize=(10, 6))

    # 计算度的频次
    def get_degree_freq(degrees):
        freq = pd.Series(degrees).value_counts().sort_index()
        return freq.index, freq.values

    orig_x, orig_y = get_degree_freq(original_vals)
    rand_x, rand_y = get_degree_freq(random_vals)
    sf_x, sf_y = get_degree_freq(scale_free_vals)

    plt.scatter(orig_x, orig_y, alpha=0.6, label='原始网络', color='skyblue')
    plt.scatter(rand_x, rand_y, alpha=0.6, label='随机网络', color='orange')
    plt.scatter(sf_x, sf_y, alpha=0.6, label='无标度网络', color='red')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('节点度')
    plt.ylabel('频次')
    plt.title('三种网络度分布对数坐标对比')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('D:/network/度分布对数坐标.png', dpi=300)
    plt.show()


# 对比三种网络的度分布
compare_degree_distributions(original_degrees, random_full_degrees, scale_free_full_degrees)