import torch
from torch import nn
import numpy as np


def print_grads_info(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.grad.shape}")

def print_grads(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.grad.abs().mean().item():.4f}")

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

"""
num_layers = 1 ,num_direction = 1
输入
- input (seq_len, batch, input_size)
- h_0 (num_layers * num_directions, batch, hidden_size)
- c_0 (num_layers * num_directions, batch, hidden_size)
输出
- output (seq_len, batch, num_directions * hidden_size)
- h_n (num_layers * num_directions, batch, hidden_size)
- c_n (num_layers * num_directions, batch, hidden_size)
"""

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(1,0,2)
        h0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(1,0,2)
        out = self.fc(out[:,-1,:])
        return out


def train(train_loader,val_loader,num_epochs,batch_size,num_steps,input_size,num_classes,device):

    model = LSTMModel(input_size, 300,num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # import ipdb;ipdb.set_trace()

            _, predicted = torch.max(outputs.data, 1)
            metric.add(loss.item() * inputs.size(0),(predicted == targets.squeeze()).sum().item(),targets.numel())

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for  i, (inputs, targets) in enumerate(val_loader):
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets.squeeze()).sum().item()
            val_acc = correct / total
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {metric[0]/metric[2]:.4f},Val Acc: {val_acc:.4f},Train Acc:{metric[1]/metric[2]:.4f}')

    import ipdb;ipdb.set_trace()

    return model
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(300,100,4)
    input = normal((1000,batch_size,300))
    output = model(input)
    print(output.shape)
    import ipdb;ipdb.set_trace()