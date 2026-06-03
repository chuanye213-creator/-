import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import jieba

# 1. 加载数据和模型
df = pd.read_csv("school_news.csv", encoding="utf-8-sig")
titles = df["标题"].dropna().astype(str).tolist()
model = Word2Vec.load("word2vec_news.model")

# 2. 句子转向量
def get_sentence_vector(sentence, model):
    words = jieba.lcut(sentence)
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

title_vectors = [get_sentence_vector(t, model) for t in titles]

# 3. 定义：指定索引找相似
def search_similar(idx, top_n=3):
    if idx < 0 or idx >= len(titles):
        print("索引错误！")
        return
    # 选指定新闻
    base_title = titles[idx]
    base_vec = title_vectors[idx]
    # 计算相似度
    similarities = []
    for i, vec in enumerate(title_vectors):
        if i != idx:  # 排除自己
            sim = np.dot(base_vec, vec) / (np.linalg.norm(base_vec) * np.linalg.norm(vec) + 1e-8)
            similarities.append((i, titles[i], round(sim, 4)))
    # 排序取Top3
    similarities.sort(key=lambda x: x[2], reverse=True)
    top3 = similarities[:top_n]
    
    # 输出实验报告格式
    print("="*80)
    print("📋 3.3 指定新闻相似搜索结果")
    print("="*80)
    print(f"> 选择的新闻索引：{idx}")
    print(f"> 新闻内容：{base_title}")
    print("> 最相似的 3 条：")
    for i, (idx_sim, title_sim, sim) in enumerate(top3):
        print(f"{i+1}. 相似度：{sim} | 新闻内容：{title_sim}")

# ========== 关键：选一个索引（比如选第10条，你可以改数字） ==========
search_similar(idx=10)  # 索引从0开始，0=第一条，10=第11条