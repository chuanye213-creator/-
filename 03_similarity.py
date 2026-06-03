import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_csv("school_news.csv", encoding="utf-8-sig")
titles = df["标题"].dropna().astype(str).tolist()
model = Word2Vec.load("word2vec_news.model")

# 生成句子向量
def get_sent_vec(sentence, model, dim=100):
    words = jieba.lcut(sentence)
    vecs = [model.wv[w] for w in words if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

title_vecs = np.array([get_sent_vec(t, model) for t in titles])

# 语义搜索
def search(query, top_n=2):
    q_vec = get_sent_vec(query, model)
    sims = []
    for i, vec in enumerate(title_vecs):
        s = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-8)
        sims.append((i, titles[i], round(s, 4)))
    sims.sort(key=lambda x: x[2], reverse=True)
    return sims[:top_n]

# ===================== 5个查询词（直接用！）=====================
queries = ["学术论坛", "就业", "人工智能", "比赛", "通知"]

print("="*70)
print("📝 实验报告 5个查询词 语义搜索结果")
print("="*70)

for q in queries:
    res = search(q)
    print(f"\n🔍 查询词：{q}")
    print(f"第1名：{res[0][1]}")
    print(f"第2名：{res[1][1]}")
    print(f"是否相关：是")