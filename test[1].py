import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer , util


#模型下载
model = SentenceTransformer('all-mpnet-base-v2')

# 编码句子
sentences = ['Python is an interpreted high-level general-purpose programming language.',
    'Python is dynamically-typed and garbage-collected.',
    'The quick brown fox jumps over the lazy dog.']

# 获得句子嵌入向量
embeddings = model.encode(sentences)
import ipdb;ipdb.set_trace()

# 打印嵌入向量
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
# Sentence: Python is an interpreted high-level general-purpose programming language.
# Embedding: [-1.17965914e-01 -4.57159936e-01 -5.87313235e-01 -2.72477478e-01 ...
# ...
# 计算相似度
sim = util.cos_sim(embeddings[0], embeddings[1])
print("{0:.4f}".format(sim.tolist()[0][0])) # 0.6445
sim = util.cos_sim(embeddings[0], embeddings[2])
print("{0:.4f}".format(sim.tolist()[0][0])) # 0.0365
