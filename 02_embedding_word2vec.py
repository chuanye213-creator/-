# 02_embedding_word2vec.py
import pandas as pd
import jieba
from gensim.models import Word2Vec
import numpy as np

# 1. 读取爬好的新闻数据
df = pd.read_csv("school_news.csv", encoding="utf-8-sig")
titles = df["标题"].dropna().tolist()
print(f"加载新闻标题数量：{len(titles)}")

# 2. 分词处理
sentences = []
for title in titles:
    words = jieba.lcut(str(title).strip())
    sentences.append(words)

# 3. 训练 Word2Vec 模型
model = Word2Vec(
    sentences=sentences,
    vector_size=100,  # 词向量维度
    window=5,
    min_count=1,
    workers=4,
    epochs=15
)

# 4. 保存模型
model.save("word2vec_news.model")
print("\n✅ Word2Vec 模型训练完成，已保存为 word2vec_news.model")

# 5. 输出关键信息（用于实验报告）
print("\n===== 词向量基本信息 =====")
print(f"词向量维度：100")
print(f"词汇表大小：{len(model.wv.index_to_key)}")
print(f"训练语料：90 条东北财经大学新闻标题")

# 6. 测试几个词
print("\n===== 相似词测试 =====")
test_words = ["学校", "学术", "会议", "学生", "教学"]
for word in test_words:
    if word in model.wv:
        sim_words = model.wv.most_similar(word, topn=3)
        print(f"{word} 相似词：{sim_words}")