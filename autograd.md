```python
import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    


```python
y = x + 2
print(y)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    


```python
print(y.grad_fn)
```

    <AddBackward0 object at 0x0000023EB6506B80>
    


```python
z = y * y * 3
out = z.mean()
print(z, out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
    


```python
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x)
```

    tensor([[-1.0000],
            [-0.9798],
            [-0.9596],
            [-0.9394],
            [-0.9192],
            [-0.8990],
            [-0.8788],
            [-0.8586],
            [-0.8384],
            [-0.8182],
            [-0.7980],
            [-0.7778],
            [-0.7576],
            [-0.7374],
            [-0.7172],
            [-0.6970],
            [-0.6768],
            [-0.6566],
            [-0.6364],
            [-0.6162],
            [-0.5960],
            [-0.5758],
            [-0.5556],
            [-0.5354],
            [-0.5152],
            [-0.4949],
            [-0.4747],
            [-0.4545],
            [-0.4343],
            [-0.4141],
            [-0.3939],
            [-0.3737],
            [-0.3535],
            [-0.3333],
            [-0.3131],
            [-0.2929],
            [-0.2727],
            [-0.2525],
            [-0.2323],
            [-0.2121],
            [-0.1919],
            [-0.1717],
            [-0.1515],
            [-0.1313],
            [-0.1111],
            [-0.0909],
            [-0.0707],
            [-0.0505],
            [-0.0303],
            [-0.0101],
            [ 0.0101],
            [ 0.0303],
            [ 0.0505],
            [ 0.0707],
            [ 0.0909],
            [ 0.1111],
            [ 0.1313],
            [ 0.1515],
            [ 0.1717],
            [ 0.1919],
            [ 0.2121],
            [ 0.2323],
            [ 0.2525],
            [ 0.2727],
            [ 0.2929],
            [ 0.3131],
            [ 0.3333],
            [ 0.3535],
            [ 0.3737],
            [ 0.3939],
            [ 0.4141],
            [ 0.4343],
            [ 0.4545],
            [ 0.4747],
            [ 0.4949],
            [ 0.5152],
            [ 0.5354],
            [ 0.5556],
            [ 0.5758],
            [ 0.5960],
            [ 0.6162],
            [ 0.6364],
            [ 0.6566],
            [ 0.6768],
            [ 0.6970],
            [ 0.7172],
            [ 0.7374],
            [ 0.7576],
            [ 0.7778],
            [ 0.7980],
            [ 0.8182],
            [ 0.8384],
            [ 0.8586],
            [ 0.8788],
            [ 0.8990],
            [ 0.9192],
            [ 0.9394],
            [ 0.9596],
            [ 0.9798],
            [ 1.0000]])
    


```python
y = x.pow(2) + 0.2*torch.rand(x.size())
print(y)
```

    tensor([[1.0784],
            [1.1479],
            [1.0546],
            [1.0607],
            [1.0387],
            [0.9370],
            [0.8019],
            [0.9239],
            [0.7247],
            [0.8227],
            [0.7182],
            [0.6199],
            [0.7664],
            [0.7176],
            [0.6005],
            [0.6004],
            [0.5699],
            [0.5733],
            [0.5863],
            [0.4447],
            [0.4722],
            [0.3736],
            [0.4752],
            [0.4678],
            [0.3650],
            [0.2645],
            [0.3004],
            [0.2838],
            [0.2650],
            [0.3185],
            [0.1734],
            [0.2651],
            [0.2805],
            [0.1938],
            [0.2889],
            [0.0948],
            [0.2662],
            [0.1389],
            [0.2253],
            [0.0765],
            [0.1797],
            [0.0368],
            [0.2028],
            [0.0453],
            [0.0445],
            [0.1368],
            [0.1276],
            [0.1653],
            [0.1094],
            [0.1516],
            [0.1640],
            [0.0646],
            [0.0328],
            [0.1674],
            [0.0430],
            [0.1760],
            [0.0906],
            [0.0271],
            [0.2104],
            [0.1463],
            [0.0744],
            [0.2221],
            [0.2463],
            [0.2249],
            [0.2037],
            [0.1004],
            [0.1667],
            [0.2048],
            [0.2943],
            [0.1959],
            [0.1805],
            [0.3207],
            [0.4008],
            [0.2745],
            [0.3392],
            [0.2687],
            [0.3435],
            [0.3982],
            [0.4960],
            [0.4583],
            [0.4287],
            [0.5026],
            [0.4528],
            [0.4792],
            [0.5671],
            [0.6866],
            [0.7369],
            [0.7057],
            [0.7792],
            [0.6811],
            [0.7489],
            [0.7764],
            [0.8698],
            [0.8316],
            [0.9135],
            [0.9295],
            [0.9579],
            [1.0983],
            [1.1251],
            [1.0983]])
    


```python
%matplotlib inline
import random
import torch
import matplotlib.pyplot as plt
```


```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) #建立一个n个w维的向量
    y = torch.matmul(X, w) + b                     #计算线性函数的值
    y += torch.normal(0, 0.01, y.shape)            #加入噪声
    return X, y.reshape((-1, 1))                   #返回结果

true_w = torch.tensor([2, -3.4])                   #y = 2*x1 - 3.4*y + 4.2
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) #生成1000个数据
```


```python
plt.scatter(features[:,1].numpy(), labels.numpy(), 2)
```




    <matplotlib.collections.PathCollection at 0x23ebd930370>




    
![png](output_8_1.png)
    



```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)         #样本的个数，这里是1000个
    indices = list(range(num_examples))  #生成0到999的序列
    random.shuffle(indices)       #打乱数据进行输出
    for i in range(0, num_examples, batch_size):  #从0到999跨度为batch_size
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_examples)]) #切割样本
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break   

```

    tensor([[ 0.6144,  1.2624],
            [ 1.2051,  0.3964],
            [-0.3858, -1.5879],
            [ 1.7007, -0.3739],
            [-0.3695, -1.0623],
            [-0.5225, -0.3030],
            [ 0.0622, -0.7604],
            [-0.2744, -1.6353],
            [-0.2362, -0.2556],
            [ 0.3125,  0.2188]]) 
     tensor([[1.1458],
            [5.2526],
            [8.8197],
            [8.8699],
            [7.0672],
            [4.1668],
            [6.9153],
            [9.2074],
            [4.5967],
            [4.0778]])
    


```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) #回归模型参数定义
b = torch.zeros(1, requires_grad=True)
def linreg(X, w, b):                                      #求值函数
    return torch.matmul(X, w) + b
```


```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')
```

    epoch1, loss0.038267
    epoch2, loss0.000147
    epoch3, loss0.000052
    


```python
print(w)
print(b)
```

    tensor([[ 2.0000],
            [-3.3997]], requires_grad=True)
    tensor([4.1996], requires_grad=True)
    


```python
from torch.utils import data
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)                    #将data_array解开入参，获取一个dataset
    return data.DataLoader(dataset, batch_size, shuffle=is_train) #批量随机获取样本
batch_size = 10
data_iter = load_array((features, labels), batch_size)            #作为迭代器，每次取小样本

next(iter(data_iter))

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))                              #线性模型，这里输入为2，输出为1

net[0].weight.data.normal_(0, 0.01)                               #设置w
net[0].bias.data.fill_(0)                                         #设置b

loss = nn.MSELoss()                                               #均方误差
trainer = torch.optim.SGD(net.parameters(), lr=0.03)              #设置递归下降

num_epochs = 3
for epoch in range(num_epochs):                                   #开始训练
    for X, y in data_iter:
        l = loss(net(X), y)                                       #获取loss
        trainer.zero_grad()                                       #grad置零 
        l.backward()                                              #计算偏导
        trainer.step()                                            #模型更新
    l = loss(net(features), labels)
    print(f'epoch{epoch + 1}, loss{l:f}')                         #显示loss
```

    epoch1, loss0.000183
    epoch2, loss0.000095
    epoch3, loss0.000095
    


```python

```
