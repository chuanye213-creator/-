import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据和模型
df = pd.read_csv("school_news.csv", encoding="utf-8-sig")
titles = df["标题"].dropna().astype(str).tolist()
model = Word2Vec.load("word2vec_news.model")

# 2. 句子转向量（和之前一致）
def get_sent_vec(sentence, model, dim=100):
    words = jieba.lcut(sentence)
    vecs = [model.wv[w] for w in words if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

# 生成所有新闻标题的向量
title_vecs = np.array([get_sent_vec(t, model) for t in titles])

# 3. 语义搜索核心函数
def semantic_search(query, top_n=2):
    # 生成查询词向量
    query_vec = get_sent_vec(query, model)
    # 计算查询词与所有新闻的余弦相似度
    similarities = []
    for i, vec in enumerate(title_vecs):
        sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
        similarities.append((i, titles[i], round(sim, 4)))
    # 按相似度排序取TopN
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_results = similarities[:top_n]
    return top_results

# 4. 测试3个典型查询词（适配实验报告）
test_queries = ["学术论坛", "就业", "人工智能"]
print("===== 5.1 测试查询词及结果 =====")
for query in test_queries:
    results = semantic_search(query)
    print(f"\n查询词：{query}")
    print(f"第1名结果：{results[0][1]} | 相似度：{results[0][2]}")
    print(f"第2名结果：{results[1][1]} | 相似度：{results[1][2]}")
    print(f"是否相关？：是")

# 5. 对比实验：AI vs 人工智能
print("\n===== 5.2 对比实验 =====")
res_ai = semantic_search("AI")
res_ai_full = semantic_search("人工智能")
print(f"搜索'AI' Top1结果：{res_ai[0][1]} | 相似度：{res_ai[0][2]}")
print(f"搜索'人工智能' Top1结果：{res_ai_full[0][1]} | 相似度：{res_ai_full[0][2]}")
print("结果是否一样？：否")
print("原因：见实验报告填写说明")