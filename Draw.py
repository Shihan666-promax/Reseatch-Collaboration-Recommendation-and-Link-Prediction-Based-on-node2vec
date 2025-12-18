import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('output/edges.csv')

# 创建图
G = nx.Graph()

# 添加边
for _, row in df.iterrows():
    G.add_edge(row['node_id_1'], row['node_id_2'], year=row['year'], weight=row['weight'])
    
print(f"总节点数: {G.number_of_nodes()}")
print(f"总边数: {G.number_of_edges()}")

# 由于图太大，我们提取一个子图
# 提取前1000条边构成的子图
subgraph_edges = list(G.edges())[:1000]
G_sub = nx.Graph()
G_sub.add_edges_from(subgraph_edges)

print(f"子图节点数: {G_sub.number_of_nodes()}")
print(f"子图边数: {G_sub.number_of_edges()}")

# 绘制子图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_sub, seed=42)  
nx.draw_networkx_nodes(G_sub, pos, node_size=50, node_color='lightblue', alpha=0.7)
nx.draw_networkx_edges(G_sub, pos, alpha=0.5, edge_color='gray')

plt.title("Subgraph Visualization (First 1000 Edges)")
plt.axis('off')
plt.tight_layout()
plt.show()