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

    word_vec['<bos>']=[0.0]*300

    def tokenlize(data):
        token = []
        for arg in data:
            arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)
            arg = arg.split(' ')
            words = []
            for word in arg:
                vector = word_vec[word]
                words.append(vector.reshape(1, -1))  # Reshape the word vector to a unified shape
            token.append(np.concatenate(words, axis=0))
        return token

    

    train_arg1_feature = tokenlize(train_arg1)  # 提取训练集中所有论元1的特征
    train_arg2_feature = tokenlize(train_arg2)
    test_arg1_feature = tokenlize(test_arg1)
    test_arg2_feature = tokenlize(test_arg2)

    def get_num_steps(percent = 0.95):
        lens = [
            len(j) for i in (train_arg1_feature, train_arg2_feature, test_arg1_feature, test_arg2_feature) for j in i
        ]
        count_dict = Counter(lens)
        sorted_counts = sorted(count_dict.items(), key=lambda x: x[0])

        total_samples = len(lens)
        threshold = total_samples * percent

        cumulative_count = 0
        for num, count in sorted_counts:
            cumulative_count += count
            if cumulative_count >= threshold:
                boundary = num
                break
        return boundary

    max_steps = get_num_steps()
    
    def pad_feature(data, max_steps):
        data_padded = []
        for feature in data:
            if len(feature) < max_steps:
                padded_feature = np.pad(feature, ((0, max_steps - len(feature)), (0, 0)), 'constant')
            else:
                padded_feature = feature[:max_steps]
            data_padded.append(padded_feature)
        return np.array(data_padded)

    # (n , max_steps ,features)
    train_feature_1 = pad_feature(train_arg1_feature, max_steps)
    train_feature_2 = pad_feature(train_arg2_feature, max_steps)
    test_feature_1 = pad_feature(test_arg1_feature, max_steps)
    test_feature_2 = pad_feature(test_arg2_feature, max_steps)

    train_feature_1 = train_feature_1[:,::-1,:]
    train_feature_2 = train_feature_2[:,::-1,:]

    train_feature = np.concatenate((train_feature_1, train_feature_2), axis=2)  # 将论元1和论元2的特征拼接
    test_feature = np.concatenate((test_feature_1, test_feature_2), axis=2)

    X_train, X_val, y_train, y_val = train_test_split(train_feature, train_label, test_size=0.2, random_state=42)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import ipdb;ipdb.set_trace()

# 转换为PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
test_feature_tensor = torch.tensor(test_feature, dtype=torch.float).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(test_feature_tensor)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


train(data_set=(train_loader,val_loader),
      num_epochs=100, batch_size=batch_size, num_steps=X_train.shape[1], input_size=X_train.shape[2], num_classes=4,device=device)

import ipdb;ipdb.set_trace()


# SVM分类
# clf = svm.SVC(decision_function_shape='ovo')

# 随机森林分类
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

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

# 计算测试集预测结果并保存
# test_pred = clf.predict(test_feature)
# with open('test_pred.txt', 'w') as f:
#     for label in test_pred:
#         f.write(str(label) + '\n')
# f.close()
