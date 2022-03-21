import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
"""
使用说明：本程序先读取cbow.py训练得到的cbow.vec文件
然后设置想要显示在热力图里的单词，比如说想显示“man”，就填到query里
最后运行程序，会自动弹出窗口，里面显示根据词向量画出来的热力图
"""
# ！！！填入想要显示的单词！！！
query = ["man", "women", "air"]


def load_pretrained(load_path):
    with open(load_path, "r") as fin:
        # Optional: depending on the specific format of pretrained vector file
        n, d = map(int, fin.readline().split())
        tokens = {}
        embeds = []
        i = 1
        for line in fin:
            line = line.rstrip().split(' ')
            token, embed = line[0], list(map(float, line[1:]))
            tokens[token] = i
            embeds.append(embed)
            i += 1
    return tokens, embeds

all_tokens, all_embeds = load_pretrained("cbow.simple.vec")
query_embeds = []
for word in query:
    query_embeds.append(all_embeds[all_tokens[word]])

# query_embeds = torch.FloatTensor(query_embeds)
# similarity = torch.cosine_similarity(query_embeds[0], query_embeds[1], dim=0)
# print('similarity', similarity)
# similarity = torch.cosine_similarity(query_embeds[0], query_embeds[2], dim=0)
# print('similarity', similarity)

query_embeds = np.array(query_embeds)
fig, ax = plt.subplots(figsize = (600,8))
# sns.heatmap(pd.DataFrame(np.round(query_embeds,2)), annot=False, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
t = sns.heatmap(np.round(query_embeds,2), annot=False, vmax=5, vmin=-5, xticklabels=False, yticklabels=True, square=True, cmap="YlGnBu")
bottom, top = t.get_ylim()
t.set_ylim(bottom + 0.5, top - 0.5)
ax.set_yticklabels(query, rotation=360, fontsize=18, horizontalalignment='right')
plt.show()