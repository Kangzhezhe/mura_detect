import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer , util

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from model import train
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter

#模型下载
phase_model = SentenceTransformer('all-mpnet-base-v2')

def get_dataset():
    # 读取数据
    train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
    test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)

    train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
    test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

    # 对类别标签编码
    class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}
    train_label = np.array([class_dict[label] for label in train_label])

    # 加载词向量文件
    word_vec = pickle.load(open('glove_300.pickle', 'rb'))

    def tokenlize(data):
        token = []
        for arg in data:
            arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)
            word = phase_model.encode(arg)
            token.append(word)
        words = np.array(token)
        return words

    train_arg1_feature = tokenlize(train_arg1)  # 提取训练集中所有论元1的特征
    train_arg2_feature = tokenlize(train_arg2)
    test_arg1_feature = tokenlize(test_arg1)
    test_arg2_feature = tokenlize(test_arg2)

    train_feature = np.concatenate((train_arg1_feature, train_arg2_feature), axis=1)  # 将论元1和论元2的特征拼接
    test_feature = np.concatenate((test_arg1_feature, test_arg2_feature), axis=1)

    X_train, X_val, y_train, y_val = train_test_split(train_feature, train_label, test_size=0.1, random_state=42)

    with open('data/data.pkl', 'wb') as file:
        pickle.dump((X_train, X_val, y_train, y_val, test_feature), file)

    return X_train, X_val, y_train, y_val, test_feature

def get_arg1(data):
    return data[..., :300]

def get_arg2(data):
    return data[..., 300:]

get_dataset()

# X (n , max_steps ,features)
# y (n)
with open('data/data.pkl', 'rb') as file:
    X_train, X_val, y_train, y_val, test_feature = pickle.load(file)

import ipdb;ipdb.set_trace()

# SVM分类
clf = svm.SVC(decision_function_shape='ovo')

# 随机森林分类
# clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 计算验证集上的Acc和F1
train_pred = clf.predict(X_val)
train_acc = accuracy_score(y_val, train_pred)
train_f1 = f1_score(y_val, train_pred, average='macro')
print(f'Val Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')

# 计算训练集上的Acc和F1
train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred, average='macro')
print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')

import ipdb;ipdb.set_trace()
# 计算测试集预测结果并保存
# test_pred = clf.predict(test_feature)
# with open('test_pred.txt', 'w') as f:
#     for label in test_pred:
#         f.write(str(label) + '\n')
# f.close()
