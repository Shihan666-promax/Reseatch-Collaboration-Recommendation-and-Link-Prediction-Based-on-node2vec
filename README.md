# Reseatch-Collaboration-Recommendation-and-Link-Prediction-Based-on-node2vec
爬取web of science 上有关LIS的论文导出，经数据清洗后得到作者列表node.csv和合作边列表edge.csv，分别利用Deepwalk和Node2vec算法对边数据进行嵌入，通过调节参数p、q控制游走策略，并利用余弦相似度进行链路预测与Top-K合作推荐。实验表明，Node2Vec在链路预测任务中AUC值均超过0.999，其中BFS策略（p=1, q=2）效果最佳；推荐结果与学者实际研究子领域高度吻合，验证了模型在合作推荐中的可行性与可解释性。
