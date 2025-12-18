import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体和图形参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_embeddings(file_path):
    #加载并处理嵌入文件
    if not os.path.exists(file_path):
        
        return None
    
    df = pd.read_csv(file_path)
    nodes = df['node'].values
    embeddings = df.drop('node', axis=1).values
    
    
    return nodes, embeddings

def perform_tsne(embeddings, n_components=3, perplexity=30, random_state=42):
    #执行t-SNE降维
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, 
                random_state=random_state,
                init='random',
                learning_rate='auto')
    
    embeddings_tsne = tsne.fit_transform(embeddings)
    return embeddings_tsne

def plot_3d_tsne(ax, embeddings_tsne, algorithm_name, color_map='viridis'):
    #绘制3D t-SNE
    # 为每个点生成颜色
    colors = np.arange(len(embeddings_tsne))
    
    scatter = ax.scatter(embeddings_tsne[:, 0], 
                       embeddings_tsne[:, 1], 
                       embeddings_tsne[:, 2],
                       c=colors, 
                       cmap=color_map,
                       alpha=0.7,
                       s=20)
    
    ax.set_title(f'{algorithm_name} - t-SNE 3D可视化', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('节点索引', rotation=270, labelpad=15)

def main():
    
    file_paths = {
        'BFS': 'output/node_embeddings_BFS.csv',
        'DFS': 'output/node_embeddings_DFS.csv', 
        'DeepWalk': 'output/node_embeddings_Deepwalk.csv'
    }
    
    
    fig = plt.figure(figsize=(21, 6))
    
    results = {}
    
    
    for i, (algo_name, file_path) in enumerate(file_paths.items(), 1):
        # 加载数据
        result = load_and_process_embeddings(file_path)
        if result is None:
            continue
            
        nodes, embeddings = result
        results[algo_name] = (nodes, embeddings)
        
        # 执行t-SNE
        
        embeddings_tsne = perform_tsne(embeddings)
        
        # 创建子图
        ax = fig.add_subplot(1, 3, i, projection='3d')
        
        # 绘制3D图
        plot_3d_tsne(ax, embeddings_tsne, algo_name)
    
    plt.tight_layout()
    plt.show()
    
    
    plt.savefig('node_embeddings_tsne_3d_comparison.png', dpi=300, bbox_inches='tight')
   

if __name__ == "__main__":
    main()