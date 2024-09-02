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
import torch
from torch import nn
from d2l import torch as d2l
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
        enc_valid_lens = []
        for feature in data:
            if len(feature) < max_steps:
                padded_feature = np.pad(feature, ((0, max_steps - len(feature)), (0, 0)), 'constant')
                enc_valid_lens.append(len(feature))
            else:
                padded_feature = feature[:max_steps]
                enc_valid_lens.append(len(feature))
            data_padded.append(padded_feature)
        return np.array(data_padded),np.array(enc_valid_lens).reshape(-1,1)

    # (n , max_steps ,features)
    train_feature_1, train_valid_lens_1 = pad_feature(train_arg1_feature, max_steps)
    train_feature_2,train_valid_lens_2= pad_feature(train_arg2_feature, max_steps)
    test_feature_1, test_valid_lens_1 = pad_feature(test_arg1_feature, max_steps)
    test_feature_2, test_valid_lens_2 = pad_feature(test_arg2_feature, max_steps)

    # train_feature_1 = train_feature_1[:,::-1,:]
    # train_feature_2 = train_feature_2[:,::-1,:]

    train_feature = np.concatenate((train_feature_1, train_feature_2), axis=2)  # 将论元1和论元2的特征拼接
    test_feature = np.concatenate((test_feature_1, test_feature_2), axis=2)

    train_len = np.concatenate((train_valid_lens_1, train_valid_lens_2), axis=1)  # 将论元1和论元2的特征拼接
    test_len = np.concatenate((test_valid_lens_1, test_valid_lens_2), axis=1)

    len_train,len_val, X_train, X_val, y_train, y_val = train_test_split(train_len,train_feature, train_label, test_size=0.2, random_state=42)

    with open('data/data_atten.pkl', 'wb') as file:
        pickle.dump((len_train,len_val,X_train, X_val, y_train, y_val, test_feature), file)

    return X_train, X_val, y_train, y_val, test_feature

embed_size = 300
def get_arg1(data):
    return data[..., :300]

def get_arg2(data):
    return data[..., 300:]

# get_dataset()

# X (n , max_steps ,features)
# y (n)
with open('data/data_atten.pkl', 'rb') as file:
    len_train,len_val,X_train, X_val, y_train, y_val, test_feature = pickle.load(file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

len_train_tensor = torch.tensor(len_train, dtype=torch.long)
len_val_tensor = torch.tensor(len_val, dtype=torch.long)
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
test_feature_tensor = torch.tensor(test_feature, dtype=torch.float)

batch_size = 256
train_arrays = (X_train_tensor, len_train_tensor, y_train_tensor)
train_iter = d2l.load_array(train_arrays, batch_size, True)

val_arrays = (X_val_tensor, len_val_tensor, y_val_tensor)


class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens,4)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = X.permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        # outputs = self.dense(torch.cat(outputs, dim=0))
        outputs = self.dense(outputs[-1])
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器

    Defined in :numref:`sec_seq2seq`"""
    def __init__(self , embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类
    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

num_hiddens, num_layers, dropout =  32, 2, 0.1

num_steps = X_train.shape[1]
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

# test model
encoder = Seq2SeqEncoder(embed_size=embed_size, num_hiddens=num_hiddens,
                             num_layers=num_layers)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(embed_size=embed_size, num_hiddens=num_hiddens,
                                  num_layers=num_layers)
decoder.eval()
# X = torch.zeros((4, 7,8), dtype=torch.float32)  # (batch_size,num_steps)

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net = EncoderDecoder(encoder, decoder)
net.apply(xavier_init_weights)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss =nn.CrossEntropyLoss()
net.train()
animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                    xlim=[10, num_epochs])

import ipdb;ipdb.set_trace()
for epoch in range(num_epochs):
    metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
    for batch in train_iter:
        optimizer.zero_grad()
        X,len,y = [x.to(device) for x in batch]
        X1 = get_arg1(X)
        X2 = get_arg2(X)
        len1 = len[:,0]
        len2 = len[:,1]
        output, state = net(X1, X2, len1)
        # output.shape,  state[0].shape,state[1].shape ,state[2].shape
        output = output.squeeze(1)
        l = loss(output, y)
        l.sum().backward()	
        optimizer.step()
        with torch.no_grad():
            metric.add(loss.item() * X.size(0),(output == y.squeeze()).sum().item(),y.numel())

# encoder = Seq2SeqEncoder(
#      embed_size, num_hiddens, num_layers, dropout)
# decoder = Seq2SeqAttentionDecoder(
#      embed_size, num_hiddens, num_layers, dropout)
# train(train_loader,val_loader,
#       num_epochs=100, batch_size=batch_size, num_steps=X_train.shape[1], input_size=X_train.shape[2], num_classes=4,device=device)



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
